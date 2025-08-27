## Befehlsübersicht (ausführen im Ordner `ESRGAN/`)

### Training auf L1 Loss - Vortraining
- Kurzes Training (Patientenweise 70/15/15-Split; random-aligned Patches im Training, volle Slices in Val/Test)
```bash
python train_ct_sr.py
```
Hinweis: Den Daten-Root-Pfad im Script anpassen (`root = ...`). GPU empfohlen.

###Feintuning - Training 2
-vollständiges Feintuning mit 3 kombinierten Loss-Funktionen
```bash
python finetune_ct_sr.py --data_root "C:\BachelorarbeitLeo\ESRGAN-Med\data\manifest-1724965242274\Spine-Mets-CT-SEG" --pretrained_g rrdb_ct_best.pth --epochs 50 --batch_size 8 --patch 256 --scale 2 --out_dir finetune_outputs --lr 1e-4 --lambda_perc 0.10 --lambda_gan 0.005 --warmup_g_only 500
```

### Evaluierung der Modellqualität (metrisch, CSV + JSON)
- Validierungsteilmenge, zufällige Slices (reproduzierbar wegen Seed), schreibt `eval_results/metrics_<split>.csv` und `summary_<split>.json`, schneller Test mit nur 3 Patienten mit jeweils 20 Bildern
```bash
python evaluate_ct_model.py --root "C:\AA_Leonard\A_Studium\Bachelorarbeit Superresolution\ESRGAN-Med\data\manifest-1724965242274\Spine-Mets-CT-SEG" --split val --model_path rrdb_ct_best.pth --max_patients 3 --max_slices_per_patient 20 --slice_sampling random --seed 42
```

- gesamter Validierungspatientendatensatz, zufällige Slices (reproduzierbar wegen Seed), schreibt `eval_results/metrics_<split>.csv` und `summary_<split>.json`
```bash
python evaluate_ct_model.py --root "C:\BachelorarbeitLeo\ESRGAN-Med\data\manifest-1724965242274\Spine-Mets-CT-SEG" --split val --model_path rrdb_ct_best.pth --seed 42
```

- Finale Testauswertung (ganzer Testsplit, GPU falls verfügbar)
```bash
python evaluate_ct_model.py --root "C:\AA_Leonard\A_Studium\Bachelorarbeit Superresolution\ESRGAN-Med\data\manifest-1724965242274\Spine-Mets-CT-SEG" --split test --model_path rrdb_ct_best.pth --output_dir eval_results --device cuda
```

- CPU erzwingen (falls keine GPU verfügbar)
```bash
python evaluate_ct_model.py --root "C:\...\Spine-Mets-CT-SEG" --split val --model_path rrdb_ct_best.pth --device cpu
```

### Visualisierung
- LR vs HR (ganze Slices, interaktive Ansicht: axial/coronal/sagittal)
```bash
python visualize_lr_hr.py --dicom_folder "C:\AA_Leonard\A_Studium\Bachelorarbeit Superresolution\ESRGAN-Med\data\manifest-1724965242274\Spine-Mets-CT-SEG\<Patientenordner>" --preset soft_tissue
```

- LR vs SR vs HR (SR via Modell, interaktiv)
```bash
python visualize_lr_sr_hr.py --dicom_folder "C:\BachelorarbeitLeo\ESRGAN-Med\data\manifest-1724965242274\Spine-Mets-CT-SEG\15041" --model_path rrdb_ct_best.pth --device cuda --preset soft_tissue
```
- LR vs SR vs HR (mit besonders kleinem Datensatz 49 Slices)
```bash
python visualize_lr_sr_hr.py --dicom_folder "C:\AA_Leonard\A_Studium\Bachelorarbeit Superresolution\ESRGAN-Med\data\klein" --model_path rrdb_ct_best.pth --device cuda --preset soft_tissue
```

- Patientenaufteilung ansehen (Metadaten: Zuordnung zur Gruppe, Id, Anzahl der Slices, MOdalität und Gerät, Schichtdicke und Pixel-Spacing), wichtig: gleichen Seed beachten:
```bash
python dump_patient_split.py --root "C:\AA_Leonard\A_Studium\Bachelorarbeit Superresolution\ESRGAN-Med\data\manifest-1724965242274\Spine-Mets-CT-SEG" --seed 42 
```

### Schnelltest (Legacy ESRGAN RGB, optional)
- Test der Original-ESRGAN-Demo (RGB, nicht CT-spezifisch)
```bash
python test.py
```
Modelle in `models/` ablegen, LR-Bilder in `LR/`.

### Hinweise
- Fensterung/Normalisierung: Werte in [-1,1]; Metriken werden auf [0,1] berechnet.
- Evaluierung: Verwende für finale Zahlen ausschließlich den Testsplit. Val nur für Selektion/Tuning.
- Outputs: Evaluierung erzeugt pro Slice eine CSV und aggregierte JSON-Zusammenfassungen.


