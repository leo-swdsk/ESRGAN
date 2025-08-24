import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from rrdb_ct_model import RRDBNet_CT
from ct_dataset_loader import CT_Dataset_SR
import os
import random
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset
from torch.cuda.amp import autocast, GradScaler
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

#test für git

# Trainingsfunktion
def train_sr_model(model, train_loader, val_loader, num_epochs=20, lr=1e-4, patience=5,
                   save_best_path="rrdb_ct_best.pth", save_last_path="rrdb_ct_last.pth",
                   plot_path="training_curve.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
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
            with autocast():
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

    # Manifest-Root mit allen Patienten-Unterordnern
    #root = r"C:\AA_Leonard\A_Studium\Bachelorarbeit Superresolution\ESRGAN-Med\data\manifest-1724965242274\Spine-Mets-CT-SEG"
    root = r"C:\BachelorarbeitLeo\ESRGAN-Med\data\manifest-1724965242274\Spine-Mets-CT-SEG" #RTX4080 Super
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
    # Train auf zufälligen, ausgerichteten Patches; Val/Test auf ganzen Slices (typisch 512x512)
    train_ds = ConcatDataset([CT_Dataset_SR(d, scale_factor=2, do_random_crop=True, hr_patch=128) for d in train_dirs])
    val_ds   = ConcatDataset([CT_Dataset_SR(d, scale_factor=2, do_random_crop=False) for d in val_dirs])
    test_ds  = ConcatDataset([CT_Dataset_SR(d, scale_factor=2, do_random_crop=False) for d in test_dirs])

    # DataLoader
    print("[Data] Creating dataloaders ...")
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True,  num_workers=4, pin_memory=True, persistent_workers=True) # spart Reinitialisiierung der WOrker zwischen den Epochen
    val_loader   = DataLoader(val_ds,   batch_size=2, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=2, shuffle=False, num_workers=2)


    # Modell initialisieren
    print("[Init] Creating model RRDBNet_CT(scale=2) ...")
    model = RRDBNet_CT(scale=2)

    # Training mit Early Stopping + Plot
    trained_model, train_losses, val_losses = train_sr_model(
        model, train_loader, val_loader,
        num_epochs=50, lr=1e-4, patience=7,
        save_best_path="rrdb_ct_best.pth",
        save_last_path="rrdb_ct_last.pth",
        plot_path="training_curve.png"
    )

    # Bestes Modell ist in rrdb_ct_best.pth gespeichert
