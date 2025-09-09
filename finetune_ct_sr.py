"""
Finetune RRDBNet (1->1 channel) on CT data with ESRGAN-style objectives.

Defaults and rationale:
- Pixel loss (L1): lambda_pix = 1.0
- Perceptual loss (VGG19 conv5_4 pre-ReLU): lambda_perc = 0.10
- Adversarial loss (Relativistic average GAN, RaGAN): lambda_gan = 0.005
- Optimizer: Adam(lr=1e-4, betas=(0.9, 0.999), weight_decay=0)
- Scheduler: MultiStepLR with milestones at 60% and 85% of total epochs (gamma=0.5)
- AMP: torch.cuda.amp with GradScaler
- EMA for generator weights with decay=0.999 (EMA used for validation/checkpointing)
- Gradient clipping for both G and D with max_norm=1.0

Conservative weights (lambda_perc=0.10, lambda_gan=0.005) are chosen for medical CT
to reduce hallucinations while still providing sharper textures than pure L1 training.
This prioritizes metric-faithful reconstructions (L1/PSNR/SSIM) over aggressively
hallucinated details.
"""

import os
import json
import math
import argparse
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch import amp as torch_amp
from torchvision import models
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
import matplotlib.pyplot as plt
from contextlib import nullcontext

from rrdb_ct_model import RRDBNet_CT
from ct_dataset_loader import CT_Dataset_SR
from evaluate_ct_model import split_patients, get_patient_dirs
from ct_sr_evaluation import evaluate_metrics


# -----------------------------
# Utility: EMA (Exponential Moving Average) hält „geglättete“ Schattenkopie der Generator‑Gewichte, ändert sich langsamer als die Live‑Gewichte
# -----------------------------
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.ema_model = type(model)() if hasattr(model, '__class__') else None
        # build a copy with same architecture - Schattenmodell
        self.ema_model = RRDBNet_CT(in_nc=1, out_nc=1, nf=64, nb=23, gc=32, scale=model.scale) #Modell wird hier nicht trainiert, sondern dient nur zum Validieren/Speichern der Gewichte
        self.ema_model.load_state_dict(model.state_dict(), strict=True)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        for k, v in self.ema_model.state_dict().items():
            if k in msd:
                self.ema_model.state_dict()[k].copy_(v.detach().mul(self.decay).add(msd[k].detach(), alpha=1.0 - self.decay))


# -----------------------------
# Discriminator: PatchGAN with SpectralNorm
# -----------------------------
def spectral_norm(module: nn.Module) -> nn.Module:
    return nn.utils.spectral_norm(module)


class PatchDiscriminatorSN(nn.Module):
    """Patch-based discriminator for 1-channel images. Outputs a score map (logits)."""
    def __init__(self, in_nc: int = 1):
        super().__init__()
        nf = 64
        layers = [
            # Schicht 1: 1 Kanal (HR-Bild) → 64 Kanäle, H×W bleibt gleich
            spectral_norm(nn.Conv2d(in_nc, nf, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            # Schicht 2: 64 Kanäle → 64 Kanäle, H×W wird halbiert (stride=2)
            spectral_norm(nn.Conv2d(nf, nf, 3, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            # Schicht 3: 64 Kanäle → 128 Kanäle, H×W bleibt halbiert
            spectral_norm(nn.Conv2d(nf, nf * 2, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            # Schicht 4: 128 Kanäle → 128 Kanäle, H×W wird weiter halbiert (stride=2)
            spectral_norm(nn.Conv2d(nf * 2, nf * 2, 3, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            # Schicht 5: 128 Kanäle → 256 Kanäle, H×W bleibt weiter halbiert
            spectral_norm(nn.Conv2d(nf * 2, nf * 4, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            # Schicht 6: 256 Kanäle → 256 Kanäle, H×W wird weiter halbiert (stride=2)
            spectral_norm(nn.Conv2d(nf * 4, nf * 4, 3, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            # Schicht 7: 256 Kanäle → 512 Kanäle, H×W bleibt weiter halbiert
            spectral_norm(nn.Conv2d(nf * 4, nf * 8, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            # Schicht 8: 512 Kanäle → 512 Kanäle, H×W wird weiter halbiert (stride=2)
            spectral_norm(nn.Conv2d(nf * 8, nf * 8, 3, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            # Schicht 9: 512 Kanäle → 1 Kanal (Logits), H×W bleibt weiter halbiert
            spectral_norm(nn.Conv2d(nf * 8, 1, 3, 1, 1))  # Output: Logits-Matrix
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------
# Perceptual Loss (VGG19 features before ReLU)
# -----------------------------
class VGG19FeatureExtractor(nn.Module):
    def __init__(self, layer: str = 'conv5_4'):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        # conv indices in torchvision VGG19 features
        conv_indices = {
            'conv1_1': 0, 'conv1_2': 2,
            'conv2_1': 5, 'conv2_2': 7,
            'conv3_1': 10, 'conv3_2': 12, 'conv3_3': 14, 'conv3_4': 16,
            'conv4_1': 19, 'conv4_2': 21, 'conv4_3': 23, 'conv4_4': 25,
            'conv5_1': 28, 'conv5_2': 30, 'conv5_3': 32, 'conv5_4': 34,
        }
        max_idx = conv_indices[layer]
        # Slice up to the convolution (pre-ReLU)
        self.features = nn.Sequential(*[vgg[i] for i in range(max_idx + 1)])
        for p in self.features.parameters():
            p.requires_grad_(False)
        self.eval()

        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def _preprocess(self, x_1ch: torch.Tensor) -> torch.Tensor:
        # Inputs are in [-1,1] with shape [B,1,H,W]; convert to 3ch and normalize
        x = (x_1ch + 1.0) * 0.5  # to [0,1]
        x = x.repeat(1, 3, 1, 1)
        x = (x - self.mean) / self.std
        return x

    def forward(self, x_1ch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self._preprocess(x_1ch)
        return self.features(x)


class PerceptualLoss(nn.Module):
    def __init__(self, layer: str = 'conv5_4', loss_fn: str = 'l1'):
        super().__init__()
        self.extractor = VGG19FeatureExtractor(layer)
        self.criterion = nn.L1Loss() if loss_fn == 'l1' else nn.MSELoss()

    def forward(self, sr_1ch: torch.Tensor, hr_1ch: torch.Tensor) -> torch.Tensor:
        self.extractor.eval()
        with torch.no_grad():
            feat_hr = self.extractor(hr_1ch)
        feat_sr = self.extractor(sr_1ch)  # gradients only through SR path
        return self.criterion(feat_sr, feat_hr)


# -----------------------------
# RaGAN Loss
# -----------------------------
class RaGANLoss:
    """
    Implements relativistic average GAN losses for discriminator and generator.
    Works with patch-based logits maps.
    """
    # Logit = roher, unnormalisierter Output vor der Sigmoid-Funktion
    def d_loss(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        real_mean = fake_logits.detach().mean()
        fake_mean = real_logits.detach().mean()
        loss_real = F.binary_cross_entropy_with_logits(real_logits - real_mean, torch.ones_like(real_logits))
        loss_fake = F.binary_cross_entropy_with_logits(fake_logits - fake_mean, torch.zeros_like(fake_logits))
        return loss_real + loss_fake

    def g_loss(self, real_logits_detached: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        real_mean = fake_logits.mean()
        fake_mean = real_logits_detached.mean()
        loss_real = F.binary_cross_entropy_with_logits(real_logits_detached - real_mean, torch.zeros_like(real_logits_detached))
        loss_fake = F.binary_cross_entropy_with_logits(fake_logits - fake_mean, torch.ones_like(fake_logits))
        return loss_real + loss_fake


# -----------------------------
# Data
# -----------------------------
def _load_split_from_json(split_json: str) -> Dict[str, list]:
    with open(split_json, 'r') as f:
        payload = json.load(f)
    result = {k: [] for k in ['train', 'val', 'test']}
    for split_name in result.keys():
        if split_name in payload.get('splits', {}):
            result[split_name] = [entry.get('path') for entry in payload['splits'][split_name] if 'path' in entry]
    return result


def build_dataloaders(root: str, scale: int, batch_size: int, patch_size: int, num_workers: int = 4, split_json: str = None,
                      degradation: str = 'blurnoise', blur_sigma_range=None, blur_kernel: int = None,
                      noise_sigma_range_norm=(0.001, 0.003), dose_factor_range=(0.25, 0.5), antialias_clean: bool = True) -> Tuple[DataLoader, DataLoader]:
    if split_json and os.path.isfile(split_json):
        print(f"[Split] Using split mapping from {split_json}")
        splits = _load_split_from_json(split_json)
        # Confirm loaded counts
        print(f"[Split] Loaded counts: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
    else:
        print("[Split] No valid split_json provided; generating 70/15/15 split with seed=42")
        splits = split_patients(root, seed=42)

    train_dirs = splits['train']
    val_dirs = splits['val']

    # Filter non-existing paths defensively (especially when loading from JSON)
    train_dirs = [p for p in train_dirs if isinstance(p, str) and os.path.isdir(p)]
    val_dirs = [p for p in val_dirs if isinstance(p, str) and os.path.isdir(p)]
    print(f"[Split] Existing dirs after filtering: train={len(train_dirs)} val={len(val_dirs)}")

    train_ds = ConcatDataset([
        CT_Dataset_SR(
            d,
            scale_factor=scale,
            do_random_crop=True,
            hr_patch=patch_size,
            degradation=degradation,
            blur_sigma_range=blur_sigma_range,
            blur_kernel=blur_kernel,
            noise_sigma_range_norm=tuple(noise_sigma_range_norm),
            dose_factor_range=tuple(dose_factor_range),
            antialias_clean=antialias_clean
        ) for d in train_dirs
    ])

    # Deterministic seeds per validation patient for reproducibility
    import hashlib
    def _fixed_seed_for_path(path: str, base: int = 42) -> int:
        h = hashlib.sha256((str(base) + '|' + os.path.normpath(path)).encode('utf-8')).hexdigest()
        return int(h[:8], 16)

    val_ds = ConcatDataset([
        CT_Dataset_SR(
            d,
            scale_factor=scale,
            do_random_crop=False,
            degradation=degradation,
            blur_sigma_range=blur_sigma_range,
            blur_kernel=blur_kernel,
            noise_sigma_range_norm=tuple(noise_sigma_range_norm),
            dose_factor_range=tuple(dose_factor_range),
            antialias_clean=antialias_clean,
            degradation_sampling='volume',
            deg_seed=_fixed_seed_for_path(d, base=42)
        ) for d in val_dirs
    ])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True, persistent_workers=False)
    # Validation on whole slices (variable sizes) → batch_size=1 to avoid collate size mismatches
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=max(1, num_workers // 2),
                            pin_memory=True, persistent_workers=(num_workers // 2) > 0)
    return train_loader, val_loader


# -----------------------------
# Models
# -----------------------------
def build_models(scale: int, pretrained_g: str = None, device: torch.device = torch.device('cpu')) -> Tuple[nn.Module, nn.Module, EMA]:
    G = RRDBNet_CT(in_nc=1, out_nc=1, scale=scale).to(device)
    if pretrained_g and os.path.isfile(pretrained_g):
        sd = torch.load(pretrained_g, map_location=device)
        if isinstance(sd, dict) and 'model' in sd and all(k in sd for k in ['epoch', 'model']):
            print(f"[Init] Detected checkpoint dict; loading weights from 'model' key: {pretrained_g}")
            sd = sd['model']
        G.load_state_dict(sd, strict=True)
        print(f"[Init] Loaded pretrained G weights from {pretrained_g}")
    else:
        print("[Init] Training G from provided weights (or randomly if path invalid)")

    D = PatchDiscriminatorSN(in_nc=1).to(device)
    ema = EMA(G, decay=0.999)
    ema.ema_model.to(device)
    return G, D, ema


# -----------------------------
# Train / Validate
# -----------------------------
def validate(G_ema: nn.Module, val_loader: DataLoader, device: torch.device) -> Dict[str, float]:
    G_ema.eval()
    metrics_accum = {k: 0.0 for k in ['MAE', 'MSE', 'RMSE', 'PSNR', 'SSIM']}
    n = 0
    use_cuda = (device.type == 'cuda')
    with torch.no_grad():
        for lr, hr in val_loader:
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)
            with torch_amp.autocast('cuda', enabled=use_cuda):
                sr = G_ema(lr)
            sr_cpu = sr.detach().cpu()
            hr_cpu = hr.detach().cpu()
            # Support batched evaluation by iterating per-sample
            if sr_cpu.ndim == 4:
                batch = sr_cpu.shape[0]
                for i in range(batch):
                    m = evaluate_metrics(sr_cpu[i], hr_cpu[i])
                    for k in metrics_accum:
                        metrics_accum[k] += float(m[k])
                n += batch
            else:
                m = evaluate_metrics(sr_cpu, hr_cpu)
                for k in metrics_accum:
                    metrics_accum[k] += float(m[k])
                n += 1
    for k in metrics_accum:
        metrics_accum[k] /= max(1, n)
    return metrics_accum


def train_one_epoch(
    G: nn.Module,
    D: nn.Module,
    ema: EMA,
    train_loader: DataLoader,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    scaler: torch_amp.GradScaler,
    perceptual_loss: PerceptualLoss,
    ragan: RaGANLoss,
    device: torch.device,
    lambda_pix: float = 1.0,
    lambda_perc: float = 0.10,
    lambda_gan: float = 0.005,
    log_interval: int = 100,
    warmup_g_only_iters: int = 0,
) -> float:
    G.train()
    D.train()
    l1 = nn.L1Loss()
    running_total = 0.0
    it = 0

    for lr, hr in train_loader:
        it += 1
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)

        # -----------------
        # Update Discriminator (skip in warmup)
        # -----------------
        if it > warmup_g_only_iters:
            optimizer_d.zero_grad(set_to_none=True)
            use_cuda = (device.type == 'cuda')
            with torch_amp.autocast('cuda', enabled=use_cuda):
                with torch.no_grad():
                    sr = G(lr)
                real_logits = D(hr)
                fake_logits = D(sr.detach())
                d_loss = ragan.d_loss(real_logits, fake_logits)
            scaler.scale(d_loss).backward()
            nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
            scaler.step(optimizer_d)
        else:
            d_loss = torch.tensor(0.0, device=device)

        # -----------------
        # Update Generator
        # -----------------
        optimizer_g.zero_grad(set_to_none=True)
        use_cuda = (device.type == 'cuda')
        with torch_amp.autocast('cuda', enabled=use_cuda):
            sr = G(lr)
            l_pix = l1(sr, hr)
            l_perc = perceptual_loss(sr, hr)
            fake_logits = D(sr)
            with torch.no_grad():
                real_logits_detached = D(hr).detach()
            l_gan = ragan.g_loss(real_logits_detached, fake_logits)
            total = lambda_pix * l_pix + lambda_perc * l_perc + lambda_gan * l_gan
        scaler.scale(total).backward()
        nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
        scaler.step(optimizer_g)
        scaler.update()

        # EMA update after G step
        ema.update(G)

        running_total += float(total.detach().cpu())

        if it % log_interval == 0:
            d_real = torch.sigmoid(real_logits.detach()).mean().item() if it > warmup_g_only_iters else 0.0
            d_fake = torch.sigmoid(fake_logits.detach()).mean().item()
            current_lr = optimizer_g.param_groups[0]['lr']
            print(f"  [Iter {it}] L_pix={l_pix.item():.4f} L_perc={l_perc.item():.4f} L_gan={l_gan.item():.4f} L_total={total.item():.4f} | D_real={d_real:.3f} D_fake={d_fake:.3f} | lr={current_lr:.6f}")

        # memory cleanup hints
        del sr, l_pix, l_perc, l_gan, total, fake_logits
        if 'real_logits' in locals():
            del real_logits
        torch.cuda.empty_cache()

    return running_total / max(1, it)


# -----------------------------
# Checkpointing / Plotting
# -----------------------------
def save_checkpoint(out_dir: str,
                    G_ema: nn.Module,
                    G_live: nn.Module,
                    D: nn.Module,
                    optimizer_g: torch.optim.Optimizer,
                    optimizer_d: torch.optim.Optimizer,
                    scaler: torch_amp.GradScaler,
                    scheduler_g: torch.optim.lr_scheduler._LRScheduler,
                    scheduler_d: torch.optim.lr_scheduler._LRScheduler,
                    epoch: int,
                    global_step: int,
                    tag: str,
                    *, metadata: dict = None,
                    ema_decay: float = 0.999):
    """Save checkpoint with complete training state (EMA/live, G/D, optimizers, schedulers, scaler)."""
    os.makedirs(out_dir, exist_ok=True)

    path_ema = os.path.join(out_dir, f'{tag}.pth')
    payload_common = {
        'epoch': int(epoch),
        'global_step': int(global_step),
        'D': D.state_dict(),
        'optimizer_g': optimizer_g.state_dict(),
        'optimizer_d': optimizer_d.state_dict(),
        'scaler': (scaler.state_dict() if (scaler is not None and isinstance(scaler, torch_amp.GradScaler)) else None),
        'scheduler_g': scheduler_g.state_dict() if scheduler_g is not None else None,
        'scheduler_d': scheduler_d.state_dict() if scheduler_d is not None else None,
        'ema_decay': float(ema_decay),
        'meta': metadata
    }
    payload_ema = {
        **payload_common,
        'model': G_ema.state_dict(),
        'ema_model': G_ema.state_dict(),
        'weights_type': 'ema'
    }
    torch.save(payload_ema, path_ema)
    print(f"[CKPT] Saved {tag} (EMA) -> {path_ema}")

    path_live = os.path.join(out_dir, f'{tag}_live.pth')
    payload_live = {
        **payload_common,
        'model': G_live.state_dict(),
        'ema_model': G_ema.state_dict(),
        'weights_type': 'live'
    }
    torch.save(payload_live, path_live)
    print(f"[CKPT] Saved {tag} (Live) -> {path_live}")


def plot_curves(history: Dict[str, list], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    # Loss curve
    try:
        plt.figure(figsize=(7,4))
        plt.plot(history['train_total'], label='Train total loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'training_curve.png'), dpi=150)
        plt.close()
    except Exception as e:
        print(f"[Plot] Could not save training_curve.png: {e}")

    # PSNR curve
    try:
        plt.figure(figsize=(7,4))
        plt.plot(history['val_psnr'], label='Val PSNR')
        plt.xlabel('Epoch')
        plt.ylabel('PSNR')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'val_psnr_curve.png'), dpi=150)
        plt.close()
    except Exception as e:
        print(f"[Plot] Could not save val_psnr_curve.png: {e}")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description='Finetune RRDB on CT with ESRGAN objectives (conservative)')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--scale', type=int, default=2, choices=[2,4])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--patch', type=int, default=192)
    parser.add_argument('--pretrained_g', type=str, default='rrdb_x2_blurnoise_best.pth')
    parser.add_argument('--out_dir', type=str, default='finetune_outputs')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint (ema/live) to resume training')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lambda_perc', type=float, default=0.10)
    parser.add_argument('--lambda_gan', type=float, default=0.005)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--warmup_g_only', type=int, default=500, help='number of iterations to train G only at start')
    parser.add_argument('--split_json', type=str, default=None, help='Path to patient split JSON (from dump_patient_split.py)')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience in epochs (None disables)')
    parser.add_argument('--early_metric', type=str, default='mae', choices=['mae','psnr'], help='Metric to monitor for early stopping')
    # Degradation options
    parser.add_argument('--degradation', type=str, default='blurnoise', choices=['clean', 'blur', 'blurnoise'], help='Degradation pipeline for LR generation')
    parser.add_argument('--blur_sigma_range', type=float, nargs=2, default=None, help='Range [lo hi] of Gaussian blur sigma; if None, defaults by scale')
    parser.add_argument('--blur_kernel', type=int, default=None, help='Explicit odd kernel size; if None, derived from sigma')
    parser.add_argument('--noise_sigma_range_norm', type=float, nargs=2, default=[0.001, 0.003], help='Gaussian noise sigma range on normalized [-1,1] image')
    parser.add_argument('--dose_factor_range', type=float, nargs=2, default=[0.25, 0.5], help='Dose factor range; noise scales ~ 1/sqrt(dose)')
    parser.add_argument('--antialias_clean', action='store_true', help='Use antialias in clean downsample')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] Using {device}")

    # Data
    train_loader, val_loader = build_dataloaders(
        root=args.data_root, scale=args.scale, batch_size=args.batch_size, patch_size=args.patch, num_workers=args.num_workers,
        split_json=args.split_json,
        degradation=args.degradation,
        blur_sigma_range=args.blur_sigma_range,
        blur_kernel=args.blur_kernel,
        noise_sigma_range_norm=args.noise_sigma_range_norm,
        dose_factor_range=args.dose_factor_range,
        antialias_clean=args.antialias_clean
    )

    # Models
    G, D, ema = build_models(scale=args.scale, pretrained_g=args.pretrained_g, device=device)
    perceptual = PerceptualLoss(layer='conv5_4').to(device)
    ragan = RaGANLoss()

    optimizer_g = Adam(G.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    optimizer_d = Adam(D.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)

    milestones = [max(1, int(args.epochs * 0.6)), max(2, int(args.epochs * 0.85))]
    scheduler_g = MultiStepLR(optimizer_g, milestones=milestones, gamma=0.5)
    scheduler_d = MultiStepLR(optimizer_d, milestones=milestones, gamma=0.5)

    # Echo CLI config
    print("[Config] data_root=", args.data_root)
    print("[Config] scale=", args.scale)
    print("[Config] epochs=", args.epochs)
    print("[Config] batch_size=", args.batch_size)
    print("[Config] patch=", args.patch)
    print("[Config] pretrained_g=", args.pretrained_g)
    print("[Config] out_dir=", args.out_dir)
    print("[Config] lr=", args.lr)
    print("[Config] lambda_perc=", args.lambda_perc)
    print("[Config] lambda_gan=", args.lambda_gan)
    print("[Config] num_workers=", args.num_workers)
    print("[Config] warmup_g_only=", args.warmup_g_only)
    print("[Config] split_json=", args.split_json)
    print("[Config] patience=", args.patience)
    print("[Config] early_metric=", args.early_metric)
    print("[Config] degradation=", args.degradation)
    print("[Config] blur_sigma_range=", args.blur_sigma_range)
    print("[Config] blur_kernel=", args.blur_kernel)
    print("[Config] noise_sigma_range_norm=", args.noise_sigma_range_norm)
    print("[Config] dose_factor_range=", args.dose_factor_range)
    print("[Config] antialias_clean=", args.antialias_clean)

    use_cuda = (device.type == 'cuda')
    scaler = torch_amp.GradScaler(enabled=use_cuda)

    history = {'train_total': [], 'val_psnr': [], 'val_mae': []}
    exp_name = f"rrdb_x{args.scale}_{args.degradation}"
    meta = {
        'experiment': exp_name,
        'scale_factor': args.scale,
        'degradation': args.degradation,
        'blur_sigma_range': args.blur_sigma_range if args.blur_sigma_range is not None else (None),
        'blur_kernel': args.blur_kernel,
        'noise_sigma_range_norm': args.noise_sigma_range_norm,
        'dose_factor_range': args.dose_factor_range,
        'notes': 'blur/noise degrader, jitter per patch (finetune)'
    }
    # write metadata JSON
    try:
        with open(f"{exp_name}.json", 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"[Meta] Wrote experiment metadata JSON -> {exp_name}.json")
    except Exception as e:
        print(f"[Meta] Could not write metadata JSON: {e}")
    best_psnr = -1e9
    best_mae = 1e9
    epochs_no_improve = 0

    # Resume training if provided
    start_epoch = 1
    iters_seen = 0
    if args.resume and os.path.isfile(args.resume):
        try:
            ckpt = torch.load(args.resume, map_location=device)
            if isinstance(ckpt, dict):
                # Prefer live weights when resuming
                if 'model' in ckpt:
                    G.load_state_dict(ckpt['model'], strict=True)
                if 'ema_model' in ckpt:
                    try:
                        ema.ema_model.load_state_dict(ckpt['ema_model'], strict=True)
                    except Exception:
                        pass
                if 'D' in ckpt:
                    D.load_state_dict(ckpt['D'], strict=True)
                if 'optimizer_g' in ckpt:
                    optimizer_g.load_state_dict(ckpt['optimizer_g'])
                if 'optimizer_d' in ckpt:
                    optimizer_d.load_state_dict(ckpt['optimizer_d'])
                if 'scheduler_g' in ckpt and ckpt['scheduler_g'] is not None:
                    try:
                        scheduler_g.load_state_dict(ckpt['scheduler_g'])
                    except Exception:
                        pass
                if 'scheduler_d' in ckpt and ckpt['scheduler_d'] is not None:
                    try:
                        scheduler_d.load_state_dict(ckpt['scheduler_d'])
                    except Exception:
                        pass
                if use_cuda and ('scaler' in ckpt and ckpt['scaler'] is not None):
                    try:
                        scaler.load_state_dict(ckpt['scaler'])
                    except Exception:
                        pass
                if 'epoch' in ckpt:
                    start_epoch = int(ckpt['epoch']) + 1
                iters_seen = int(ckpt.get('global_step', 0))
                best_psnr = float(ckpt.get('best_psnr', best_psnr))
                best_mae = float(ckpt.get('best_mae', best_mae))
                print(f"[Resume] Resumed from {args.resume} at epoch={start_epoch} global_step={iters_seen}")
        except Exception as e:
            print(f"[Resume] Failed to load resume checkpoint: {e}")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"[Epoch {epoch}/{args.epochs}] Starting ...")
        # Per-epoch resample for training datasets (volume-wise)
        subdatasets = getattr(train_loader.dataset, 'datasets', None)
        if subdatasets is not None:
            for ds in subdatasets:
                if hasattr(ds, 'resample_volume_params'):
                    ds.resample_volume_params(epoch_seed=epoch)
            print(f"[Deg-Resample] Training volumes resampled for epoch {epoch}")
        else:
            if hasattr(train_loader.dataset, 'resample_volume_params'):
                train_loader.dataset.resample_volume_params(epoch_seed=epoch)
                print(f"[Deg-Resample] Training volumes resampled for epoch {epoch}")
        warmup_iters_remaining = max(0, args.warmup_g_only - iters_seen)
        avg_train_total = train_one_epoch(
            G, D, ema, train_loader, optimizer_g, optimizer_d, scaler, perceptual, ragan, device,
            lambda_pix=1.0, lambda_perc=args.lambda_perc, lambda_gan=args.lambda_gan,
            log_interval=100, warmup_g_only_iters=warmup_iters_remaining
        )
        history['train_total'].append(avg_train_total)

        # We can estimate how many iters happened
        iters_seen += len(train_loader)

        # Scheduler steps per epoch
        scheduler_g.step()
        scheduler_d.step()

        # Validation using EMA
        val_metrics = validate(ema.ema_model, val_loader, device)
        history['val_psnr'].append(val_metrics['PSNR'])
        history['val_mae'].append(val_metrics['MAE'])
        print(f"[Val] MAE={val_metrics['MAE']:.6f} MSE={val_metrics['MSE']:.6f} RMSE={val_metrics['RMSE']:.6f} PSNR={val_metrics['PSNR']:.4f} SSIM={val_metrics['SSIM']:.4f}")

        # Checkpoints
        improved = False
        if val_metrics['PSNR'] > best_psnr + 1e-6:
            best_psnr = val_metrics['PSNR']
            improved = True
        if val_metrics['MAE'] < best_mae - 1e-6:
            best_mae = val_metrics['MAE']
            improved = True
        if improved:
            save_checkpoint(
                args.out_dir, ema.ema_model, G, D,
                optimizer_g, optimizer_d, scaler,
                scheduler_g, scheduler_d,
                epoch, iters_seen,
                tag='best', metadata=meta, ema_decay=ema.decay
            )
        save_checkpoint(
            args.out_dir, ema.ema_model, G, D,
            optimizer_g, optimizer_d, scaler,
            scheduler_g, scheduler_d,
            epoch, iters_seen,
            tag='last', metadata=meta, ema_decay=ema.decay
        )

        # Early stopping on selected metric
        if args.patience is not None:
            if args.early_metric == 'mae':
                had_improve = (val_metrics['MAE'] <= best_mae + 1e-6)
            else:
                had_improve = (val_metrics['PSNR'] >= best_psnr - 1e-6)
            if had_improve:
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"[EarlyStop] No improvement for {epochs_no_improve}/{args.patience} epochs on {args.early_metric.upper()}")
                if epochs_no_improve >= args.patience:
                    print("[EarlyStop] Patience reached. Stopping training early.")
                    break

    # Plots
    plot_curves(history, args.out_dir)


if __name__ == '__main__':
    main()


