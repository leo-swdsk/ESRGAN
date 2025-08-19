import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from rrdb_ct_model import RRDBNet_CT
from ct_dataset_loader import CT_Dataset_SR

# Trainingsfunktion
def train_sr_model(model, dataloader, num_epochs=5, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for i, (lr_imgs, hr_imgs) in enumerate(dataloader):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            preds = model(lr_imgs)
            loss = criterion(preds, hr_imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")

    return model

# Startpunkt
if __name__ == "__main__":
    # === TEST-DATENPFAD ===
    dicom_folder = r"C:\AA_Leonard\A_Studium\Bachelorarbeit Superresolution\ESRGAN-Med\data\manifest-1724965242274\Spine-Mets-CT-SEG\10352\12-03-2011-NA-SpineSPINEBONESBRT Adult-55418\4.000000-SKINTOSKINSIM0.5MM10352a iMAR-32611"

    # Dataset vorbereiten
    #begrenzt auf 20 Slices dataset = CT_Dataset_SR(dicom_folder, max_slices=20)
    dataset = CT_Dataset_SR(dicom_folder, scale_factor=2)
    print(f"Anzahl Datensätze: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Modell initialisieren
    model = RRDBNet_CT(scale=2)

    # Training starten
    trained_model = train_sr_model(model, dataloader, num_epochs=3)

    # Modell speichern
    torch.save(trained_model.state_dict(), "rrdb_ct_trained.pth")
