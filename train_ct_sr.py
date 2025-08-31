import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from rrdb_ct_model import RRDBNet_CT
from ct_dataset_loader import CT_Dataset_SR
import os
import random
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset
from torch import amp

#AMP (Automatic Mixed Precision) wird genutzt --> Operationen laufen intern in float16 und nicht 32, was Speicher spart
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for lr_imgs, hr_imgs in dataloader:
            lr_imgs = lr_imgs.to(device, non_blocking=True) # Pytorch kann Transfer asynchron starten-->während die GPU noch rechnet, kann schon der nächste Batch kopiert werde 
            hr_imgs = hr_imgs.to(device, non_blocking=True)
            with amp.autocast('cuda'): #Tensor Cores rechnen in float16
                preds = model(lr_imgs)
                loss = criterion(preds, hr_imgs)
            total_loss += loss.item()
    model.train()
    return total_loss / max(1, len(dataloader))

# Trainingsfunktion
def train_sr_model(model, train_loader, val_loader, num_epochs=20, lr=1e-4, patience=5,
                   save_best_path="rrdb_ct_best.pth", save_last_path="rrdb_ct_last.pth",
                   plot_path="training_curve.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr) # Betas: 0.9, 0.999 (voreingestellter Standard)
    criterion = nn.L1Loss()  

    best_val = float("inf")
    epochs_no_improve = 0
    train_losses, val_losses = [], []

    scaler = amp.GradScaler('cuda')
    for epoch in range(num_epochs):
        model.train()
        total_train = 0.0
        for batch_idx, (lr_imgs, hr_imgs) in enumerate(train_loader, start=1):
            lr_imgs = lr_imgs.to(device, non_blocking=True)
            hr_imgs = hr_imgs.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True) # Gradienten nicht 0, sondern None -> weniger Speicherzugriffe
            with amp.autocast('cuda'):
                preds = model(lr_imgs)
                loss = criterion(preds, hr_imgs)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train += loss.item()
            if batch_idx % 50 == 0:
                print(f"  [Batch {batch_idx}/{len(train_loader)}] train L1={loss.item():.5f}")
            # (optional) Speicher aufräumen:
            del preds, loss
        torch.cuda.empty_cache()

        avg_train = total_train / max(1, len(train_loader))
        avg_val = validate(model, val_loader, criterion, device)
        train_losses.append(avg_train)
        val_losses.append(avg_val)

        print(f"[Train] Epoch {epoch+1} done | train L1: {avg_train:.6f} | val L1: {avg_val:.6f}")

        # Early Stopping + Best speichern
        if avg_val < best_val - 1e-6:
            best_val = avg_val
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_best_path)
            print(f"[Train] Saved new best model -> {save_best_path}")
        else:
            epochs_no_improve += 1
            print(f"[Train] No val improvement ({epochs_no_improve}/{patience})")
            if epochs_no_improve >= patience:
                print(f"[Train] Early stopping. Best val: {best_val:.6f}")
                break

    # Letztes Modell speichern
    torch.save(model.state_dict(), save_last_path)
    print(f"[Train] Saved last model -> {save_last_path}")

    # Plot speichern
    try:
        plt.figure(figsize=(7,4))
        plt.plot(range(1, len(train_losses)+1), train_losses, label='Train loss (L1)')
        plt.plot(range(1, len(val_losses)+1), val_losses, label='Val loss (L1)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"[Train] Saved training curve -> {plot_path}")
    except Exception as e:
        print(f"[Train] Could not save plot: {e}")

    return model, train_losses, val_losses


# Startpunkt
if __name__ == "__main__":
    random.seed(42)

    parser = argparse.ArgumentParser(description='Train RRDBNet_CT on CT super-resolution with L1 loss (pretraining)')
    default_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'preprocessed_data')
    parser.add_argument('--data_root', type=str, default=default_root, help='Root with patient subfolders (default: ESRGAN/preprocessed_data)')
    parser.add_argument('--scale', type=int, default=2, help='Upscaling factor (must divide patch_size)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=10, help='Training batch size')
    parser.add_argument('--patch_size', type=int, default=192, help='HR patch size (must be divisible by scale)')
    parser.add_argument('--patience', type=int, default=7, help='Early stopping patience (epochs)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Dataloader workers for training')
    args = parser.parse_args()

    if args.patch_size % args.scale != 0:
        raise ValueError(f"patch_size ({args.patch_size}) must be divisible by scale ({args.scale})")

    print("[Args] Training configuration:")
    print(f"  data_root   : {args.data_root}")
    print(f"  scale       : {args.scale}")
    print(f"  epochs      : {args.epochs}")
    print(f"  batch_size  : {args.batch_size}")
    print(f"  patch_size  : {args.patch_size}")
    print(f"  patience    : {args.patience}")
    print(f"  lr          : {args.lr}")
    print(f"  num_workers : {args.num_workers}")

    root = args.data_root
    patient_dirs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if len(patient_dirs) == 0:
        raise RuntimeError(f"No patient directories found under {root}")

    random.shuffle(patient_dirs)
    n = len(patient_dirs)
    # 70/15/15 patient-wise split using deterministic index cutoffs
    train_cut = int(0.70 * n)
    val_cut = int(0.85 * n)
    train_dirs = patient_dirs[: train_cut]
    val_dirs   = patient_dirs[train_cut: val_cut]
    test_dirs  = patient_dirs[val_cut:]

    print(f"[Split] Patients total={n} | train={len(train_dirs)} | val={len(val_dirs)} | test={len(test_dirs)}")

    # Datasets zusammenfassen
    print("[Data] Building datasets ...")
    # Train auf zufälligen, ausgerichteten Patches; Val/Test auf ganzen Slices
    train_ds = ConcatDataset([CT_Dataset_SR(d, scale_factor=args.scale, do_random_crop=True, hr_patch=args.patch_size, normalization='global', hu_clip=(-1000, 2000)) for d in train_dirs])
    val_ds   = ConcatDataset([CT_Dataset_SR(d, scale_factor=args.scale, do_random_crop=False, normalization='global', hu_clip=(-1000, 2000)) for d in val_dirs])
    test_ds  = ConcatDataset([CT_Dataset_SR(d, scale_factor=args.scale, do_random_crop=False, normalization='global', hu_clip=(-1000, 2000)) for d in test_dirs])

    # DataLoader
    print("[Data] Creating dataloaders ...")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=True, persistent_workers=True if args.num_workers > 0 else False)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=max(1, args.num_workers//2))
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False, num_workers=max(1, args.num_workers//2))

    # Modell initialisieren
    print(f"[Init] Creating model RRDBNet_CT(scale={args.scale}) ...")
    model = RRDBNet_CT(scale=args.scale)

    # Training mit Early Stopping + Plot
    trained_model, train_losses, val_losses = train_sr_model(
        model, train_loader, val_loader,
        num_epochs=args.epochs, lr=args.lr, patience=args.patience,
        save_best_path="rrdb_ct_best.pth",
        save_last_path="rrdb_ct_last.pth",
        plot_path="training_curve.png"
    )

    # Bestes Modell ist in rrdb_ct_best.pth gespeichert
