import torch
from rrdb_ct_model import RRDBNet_CT
from ct_dataset_loader import CT_Dataset_SR
from ct_sr_evaluation import compare_methods

# Modell laden
model = RRDBNet_CT()
model.load_state_dict(torch.load("rrdb_ct_best.pth"))
model.eval()

# Gleicher Datensatz wie fürs Training
dataset = CT_Dataset_SR(
    r"C:\AA_Leonard\A_Studium\Bachelorarbeit Superresolution\ESRGAN-Med\data\manifest-1724965242274\Spine-Mets-CT-SEG",
    max_slices=100, 
    scale_factor=2,
    do_random_crop=False
)

# Teste die ersten 5 Bilder
for i in range(len(dataset)):
    lr, hr = dataset[i]
    result = compare_methods(lr, hr, model)
    print(f"Slice {i+1}")
    for method, metrics in result.items():
        print(f"{method}: {metrics}")
