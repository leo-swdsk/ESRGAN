## Befehlsübersicht (ausführen im Ordner `ESRGAN/`)

### Preprocessing – CT In‑Plane Resampling (einmalig)
- Resampling aller CT‑DICOM‑Slices auf einheitliches Pixel‑Spacing von 0.8 mm (in‑plane), lineare Interpolation.
- Output liegt standardmäßig in `ESRGAN/preprocessed_data` (Unterordnerstruktur pro Patient bleibt erhalten, Patientenordner erhalten Suffix `pp`).
- Logs: `preprocessing_log.csv` und `preprocessing_log.json` werden NUR im Output‑Ordner geschrieben.
```bash
python preprocess_resample_ct.py --root "C:\BachelorarbeitLeo\ESRGAN-Med\data\manifest-1724965242274\Spine-Mets-CT-SEG"
```
- Optionalen Output‑Pfad angeben:
```bash
python preprocess_resample_ct.py --root "C:\...\Spine-Mets-CT-SEG" --out_dir "D:\SR\preprocessed_data" --target_spacing 0.8
```
- Hinweise:
  - Nur CT‑Image‑DICOMs werden verarbeitet (SEG etc. werden ignoriert).
  - DICOM‑Header wird angepasst: `PixelSpacing`, `Rows`, `Columns`; neue `SOPInstanceUID` pro Slice.
  - VR‑Ambiguitäten bei `Smallest/LargestImagePixelValue` werden korrekt gesetzt (US/SS) oder entfernt.

### Training auf L1 Loss - Vortraining
- Kurzes Training (Patientenweise 70/15/15-Split; random-aligned Patches im Training, volle Slices in Val/Test)
```bash
python train_ct_sr.py
```
Hinweis: Den Daten-Root-Pfad im Script anpassen (`root = ...`). GPU empfohlen.

###Feintuning - Training 2
-vollständiges Feintuning mit 3 kombinierten Loss-Funktionen, bei Angabe von Split auf gleichen Daten wie Vortraining; vernünftige VRAM Auslastung, etwa 11/16 GB bei Training, 3,5 bei Validierung
```bash
python finetune_ct_sr.py --data_root "C:\BachelorarbeitLeo\ESRGAN-Med\data\manifest-1724965242274\Spine-Mets-CT-SEG" --pretrained_g rrdb_ct_best.pth --epochs 30 --batch_size 20 --patch 128 --scale 2 --out_dir finetune_outputs --lr 1e-4 --lambda_perc 0.10 --lambda_gan 0.005 --warmup_g_only 500 --split_json ESRGAN\splits\patient_split_seed42.json --patience 3 --early_metric mae
```

### Evaluierung der Modellqualität (metrisch, CSV + JSON)
- Validierungsteilmenge, zufällige Slices (reproduzierbar wegen Seed), schreibt `eval_results/metrics_<split>.csv` und `summary_<split>.json`, schneller Test mit nur 3 Patienten mit jeweils 20 Bildern; AUF DEM VORTRAININGS-MODELL
```bash
python evaluate_ct_model.py --root "C:\AA_Leonard\A_Studium\Bachelorarbeit Superresolution\ESRGAN-Med\data\manifest-1724965242274\Spine-Mets-CT-SEG" --split val --model_path rrdb_ct_best.pth --max_patients 3 --max_slices_per_patient 20 --slice_sampling random --seed 42
```

- gesamter Validierungspatientendatensatz, zufällige Slices (reproduzierbar wegen Seed), schreibt `eval_results/metrics_<split>.csv` und `summary_<split>.json`
```bash
python evaluate_ct_model.py --root "C:\BachelorarbeitLeo\ESRGAN-Med\data\manifest-1724965242274\Spine-Mets-CT-SEG" --split val --model_path rrdb_ct_best.pth --seed 42
```

- Evaluierung des finetuneten Modells (EMA‑Best) – Dateien enthalten den Modellnamen und überschreiben nichts
```bash
python evaluate_ct_model.py --root "C:\BachelorarbeitLeo\ESRGAN-Med\data\manifest-1724965242274\Spine-Mets-CT-SEG" --split val --model_path finetune_outputs\best.pth --device cuda
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

- LR vs SR vs HR (SR via Modell, interaktiv), ABER MIT DEM FINETUNING-MODELL
```bash
python visualize_lr_sr_hr.py --dicom_folder "C:\BachelorarbeitLeo\ESRGAN-Med\data\manifest-1724965242274\Spine-Mets-CT-SEG\15041" --model_path finetune_outputs\best.pth --device cuda --preset soft_tissue
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


