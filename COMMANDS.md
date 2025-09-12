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
- Kurzes Training (Patientenweise 70/15/15-Split; random-aligned Patches im Training, volle Slices in Val/Test). Standardmäßig wird `ESRGAN/preprocessed_data` verwendet.
```bash
python train_ct_sr.py
```
- Kurze Trainingsprompts je Degradationsmodus (nur wesentliche Flags):

  - Clean (nur Downsample; optional sanfter mit Antialias):
  ```bash
  python train_ct_sr.py --degradation clean --antialias_clean
  ```

  - Blur (Gauss-Blur → Downsample, kein Noise; Defaults für σ werden automatisch gewählt):
  ```bash
  python train_ct_sr.py --degradation blur
  ```

  - Blurnoise (Gauss-Blur → Downsample → dosis-skaliertes Rauschen):
  ```bash
  python train_ct_sr.py --degradation blurnoise
  ```
- Hinweise:
  - `patch_size` muss durch `scale` teilbar sein.
  - Für Verwendung ohne Preprocessing: `--data_root` auf den Rohdaten-Root setzen.
  - GPU empfohlen.

### Feintuning - Training 2
- vollständiges Feintuning mit 3 kombinierten Loss-Funktionen, bei Angabe von Split auf gleichen Daten wie Vortraining; vernünftige VRAM Auslastung, etwa 11/16 GB bei Training, 3,5 bei Validierung
```bash
python finetune_ct_sr.py --data_root "C:\BachelorarbeitLeo\ESRGAN-Med\ESRGAN\preprocessed_data" --pretrained_g rrdb_ct_best.pth --epochs 30 --batch_size 20 --patch 128 --scale 2 --out_dir finetune_outputs --lr 1e-4 --lambda_perc 0.10 --lambda_gan 0.005 --warmup_g_only 500 --split_json ESRGAN\splits\patient_split_seed42.json --patience 4 --early_metric mae --degradation blurnoise --dose_factor_range 0.25 0.5
```

### Evaluierung der Modellqualität (metrisch, CSV + JSON)
- Standard: globale Normalisierung (HU‑Clip [-1000, 2000]) + Degradation `blurnoise`. Dateinamen enthalten Modell und Normalisierungs‑Tag.

- Was wird gerechnet?
  - Es werden ganze Slices ausgewertet (kein Random‑Crop). Falls Dimensionen nicht exakt passen (ungerade Pixelzahl o. ä.), wird zentriert so zugeschnitten, dass Referenz (HR) und Rekonstruktion (SR/Interpolation) dieselbe Größe haben.
  - Metriken pro Slice: MSE, RMSE, MAE, PSNR, SSIM, LPIPS, PI. LPIPS und PI folgen derselben robusten Implementierung wie in `visualize_lr_sr_hr.py`:
    - LPIPS: bevorzugt `pyiqa` (`create_metric('lpips')`), Fallback auf `lpips`‑Paket. Graustufen werden intern auf 3 Kanäle gemappt.
    - PI: PI = 0.5*((10 − Ma) + NIQE). Bevorzugt `pyiqa`‑Metriken `nrqm` (Ma) und `niqe`. Falls nur NIQE verfügbar ist, wird NIQE als Surrogat‑PI ausgegeben, damit keine `nan` entstehen.
  - CSV enthält alle Slice‑Ergebnisse; JSON fasst global pro Slice, pro Patient und über Patienten gemittelt zusammen.

- Beispiel (Global HU‑Clip, Val-Split, schneller Lauf):
```bash
python evaluate_ct_model.py \
  --root "preprocessed_data" \
  --split val \
  --model_path finetune_outputs\best.pth \
  --hu_clip -1000 2000 \
  --max_patients 3 --max_slices_per_patient 20 --slice_sampling random \
  --degradation blurnoise --dose_factor_range 0.25 0.5 --seed 42
```

- Beispiel (Soft‑Tissue‑Fensterung, kompletter Val‑Split):
```bash
python evaluate_ct_model.py \
  --root "preprocessed_data" \
  --split val \
  --model_path finetune_outputs\best.pth \
  --preset soft_tissue --window_center 40 --window_width 400 \
  --degradation blurnoise --seed 42
```

- Ausgabe-Dateien (Beispielnamen):
  - Global: `metrics_val_rrdb_x2_blurnoise_best__globalHU_-1000_2000.csv`, `summary_val_rrdb_x2_blurnoise_best__globalHU_-1000_2000.json`
  - Soft Tissue: `metrics_val_rrdb_x2_blurnoise_best__preset-soft_tissue.csv`, `summary_val_rrdb_x2_blurnoise_best__preset-soft_tissue.json`

- CPU erzwingen (falls keine GPU verfügbar):
```bash
python evaluate_ct_model.py --root "preprocessed_data" --split val --model_path rrdb_x2_blurnoise_best.pth --device cpu
```

### Evaluierung – Testsplit reproduzierbar (per-volume Sampling, neue Run-Ordnerstruktur)
- Führt die Evaluation auf dem Testsplit mit globaler HU-Normalisierung [-1000, 2000] aus.
- Nutzt dein trainiertes Modell unter `runs\rrdb_x2_blurnoise_20250912-114004\best.pth`.
- Reproduzierbarkeit: feste Seeds für Slice-Sampling (`--seed 42`) und Degradations-Sampling per Volume (`--degradation_sampling volume --deg_seed 42`).
- Degradationsmodus `blurnoise` mit standardisierten Parametern, identisch zur Trainingskonfiguration.
- Outputs: Es wird ein Run-Verzeichnis unter `eval_results/` erstellt (mit Zeitstempel), z. B. `eval_results/test__best__globalHU_-1000_2000_on_test_set__YYYYMMDD-HHMMSS/` mit kurzen Dateinamen:
  - `metrics.csv`, `summary.json`, sowie Plot-PNGs unter `plots/`.
```bash
python evaluate_ct_model.py --root "preprocessed_data" --split test --model_path "runs\rrdb_x2_blurnoise_20250912-114004\best.pth" --device cuda --scale 2 --hu_clip -1000 2000 --degradation blurnoise --noise_sigma_range_norm 0.001 0.003 --dose_factor_range 0.25 0.5 --degradation_sampling volume --deg_seed 42 --seed 42
```
Dieser obere Befehl ist gleichwertig zu: 
```bash
python evaluate_ct_model.py --root "preprocessed_data" --split test --model_path "runs\rrdb_x2_blurnoise_20250912-114004\best.pth"
```

 Unterschied: seed vs. deg_seed
 - seed: steuert die allgemeine Auswahllogik, z. B. die Zufallsauswahl von Slices, falls du Limits setzt oder slice_sampling=random nutzt. Hat keinen Effekt, wenn alle Slices ausgewertet werden und keine weitere zufällige Auswahl nötig ist.
 - deg_seed: steuert ausschließlich die Zufallsziehung der Degradationsparameter (Blur/Noise) für die LR-Erzeugung. In Kombination mit --degradation_sampling (z. B. volume oder det-slice) sorgt er für reproduzierbare Blur-/Noise-Samples.


- Abhängigkeiten für LPIPS/PI:
  - Empfohlen: `pip install pyiqa` (liefert LPIPS, NIQE und Ma/NRQM). Alternativ für LPIPS: `pip install lpips`.
  - Falls `pyiqa` ohne Ma/NRQM installiert ist, wird PI ohne Ma berechnet (Surrogat = NIQE), damit keine `nan`‑Werte entstehen.
  - Alle Metriken werden auf Bildwerten in [0,1] berechnet; die Eingaben werden intern korrekt aus [-1,1] umskaliert.

- Hinweis zu vollen Slices und Zuschnitt:
  - Für die Evaluierung werden die HR‑Slices (und die SR/Interpolation) ggf. zentriert zugeschnitten, um exakte Größenübereinstimmung zu garantieren. Das ist bei ungeraden Dimensionen oder leichten Abweichungen normal und beabsichtigt.

### Visualisierung
- LR vs HR (ganze Slices, interaktive Ansicht: axial/coronal/sagittal)
```bash
python visualize_lr_hr.py --dicom_folder "C:\BachelorarbeitLeo\ESRGAN-Med\ESRGAN\preprocessed_data\15041pp" --preset soft_tissue --degradation blurnoise --dose_factor_range 0.25 0.5
```

#### Viewer – identisch zur Evaluation (per summary.json)
- Nutzt die per Patient in der Evaluation gezogenen Degradationswerte aus `summary.json` und erzwingt sie als Punktintervalle.
- Ergebnis: Die Viewer-Metriken pro Slice entsprechen den CSV-Werten (bei gleicher Fensterung, z. B. `--preset global`).
```bash
python visualize_lr_sr_hr.py \
  --dicom_folder "preprocessed_data\14655pp" \
  --model_path "runs\rrdb_x2_blurnoise_20250912-114004\best.pth" \
  --device cuda --scale 2 --preset global \
  --degradation blurnoise --degradation_sampling volume --deg_seed_mode per_patient --deg_seed 42 \
  --eval_summary "eval_results\test__best__globalHU_-1000_2000_on_test_set__YYYYMMDD-HHMMSS\summary.json"
```
- Hinweise:
  - `--eval_summary` entnimmt aus `patient_to_degradation_sampled` die Werte für den angegebenen Patienten und setzt `sigma`, `kernel`, `noise_sigma`, `dose` fix.
  - Passe `--dicom_folder` (z. B. `14655pp`) und den Pfad zur `summary.json` an deinen Lauf an.
  - Für andere Patienten einfach den jeweiligen Ordner wählen; die Werte werden automatisch übernommen.

- LR vs SR vs HR (SR via Modell, interaktiv)
```bash
python visualize_lr_sr_hr.py --dicom_folder "C:\BachelorarbeitLeo\ESRGAN-Med\data\manifest-1724965242274\Spine-Mets-CT-SEG\15041" --model_path rrdb_x2_blurnoise_best.pth --device cuda --preset soft_tissue --degradation blurnoise --dose_factor_range 0.25 0.5
```

- LR vs SR vs HR (SR via Modell, interaktiv) auf den homogenisierten Daten mit Pixel Spacing von 0.8 mm
```bash
python visualize_lr_sr_hr.py --dicom_folder "C:\BachelorarbeitLeo\ESRGAN-Med\ESRGAN\preprocessed_data\15040pp" --model_path rrdb_ct_best.pth --device cuda --preset soft_tiss
ue
```

- LR vs SR vs HR (SR via Modell, interaktiv), ABER MIT DEM FINETUNING-MODELL
```bash
python visualize_lr_sr_hr.py --dicom_folder "C:\BachelorarbeitLeo\ESRGAN-Med\data\manifest-1724965242274\Spine-Mets-CT-SEG\15041" --model_path finetune_outputs\best.pth --device cuda --preset soft_tissue --degradation blurnoise
```

- LR vs SR vs HR (mit besonders kleinem Datensatz 49 Slices)
```bash
python visualize_lr_sr_hr.py --dicom_folder "C:\AA_Leonard\A_Studium\Bachelorarbeit Superresolution\ESRGAN-Med\data\klein" --model_path rrdb_ct_best.pth --device cuda --preset soft_tissue
```

- Patientenaufteilung ansehen (Metadaten: Zuordnung zur Gruppe, Id, Anzahl der Slices, MOdalität und Gerät, Schichtdicke und Pixel-Spacing), wichtig: gleichen Seed beachten:
```bash
python dump_patient_split.py --root "C:\AA_Leonard\A_Studium\Bachelorarbeit Superresolution\ESRGAN-Med\data\manifest-1724965242274\Spine-Mets-CT-SEG" --seed 42 
```

### Hinweise
- Fensterung/Normalisierung: Werte in [-1,1]; Metriken werden auf [0,1] berechnet.
- Evaluierung: Verwende für finale Zahlen ausschließlich den Testsplit. Val nur für Selektion/Tuning.
- Outputs: Evaluierung erzeugt pro Slice eine CSV und aggregierte JSON-Zusammenfassungen.




### Degradationspipeline (Blur → Downsample → Noise)

Standard: `--degradation blurnoise`. Ziel ist die realistischere Simulation einer Dosisreduktion (mehr Rauschen, etwas unschärfer), auf der das Modell trainiert, validiert, evaluiert und visualisiert wird.

- Modi:
  - **clean**: nur Downsample; optional mit `--antialias_clean` (Default: aus; empfohlen für „sanftes“ Downsampling). Kein Blur, kein Noise.
  - **blur**: Gauss-Blur (σ abhängig von Scale bzw. per Range), dann Downsample mit `antialias=False`, kein Noise.
  - **blurnoise** (Default): Gauss-Blur → Downsample (`antialias=False`) → Gaussian-Noise mit dosisbasierter Skalierung.

- Konsistenz über Skripte:
  - Verwende dieselben Degradations-Flags in `train_ct_sr.py`, `finetune_ct_sr.py`, `evaluate_ct_model.py`, `visualize_lr_hr.py`, `visualize_lr_sr_hr.py`.
  - Training/Fine-Tuning speichern eine `<experiment>.json` und betten die Meta-Daten in die `.pth`-Checkpoints ein (Schlüssel `meta`). Die Evaluierung liest die Flags aktuell nicht aus dem Checkpoint – setze dort dieselben Parameter per CLI.
  - Dateinamen: `rrdb_x{scale}_{degradation}_best.pth`, `rrdb_x{scale}_{degradation}_last.pth` usw.

- Parameter und Wirkung im Bild:
  - `--degradation {clean,blur,blurnoise}`: wählt die Pipeline.
  - `--blur_sigma_range lo hi`: Gauss‑σ in der Normalisierung [-1,1]. Größeres σ ⇒ stärkere Unschärfe. Defaults (falls nicht gesetzt): bei ×2 ca. 0.8±0.1, bei ×4 ca. 1.2±0.15.
  - `--blur_kernel k`: Fixiert den (ungeraden) Kernel; ohne Angabe wird k ≈ 6·σ (auf nächste ungerade Zahl) verwendet.
  - `--noise_sigma_range_norm lo hi`: Rausch‑Std in der Normalisierung [-1,1]. Höher ⇒ sichtbarer „Korn“. Richtwert: 0.003 ≈ ca. 10 HU.
  - `--dose_factor_range lo hi`: Skaliert das Rauschen mit 1/√dose. Niedrige Dosis (z. B. 0.25) ⇒ mehr Rauschen.
  - `--antialias_clean`: Nur relevant für `clean`. Aktiviert Antialias beim Downsample (bei `blur`/`blurnoise` ist es bewusst deaktiviert, da der Blur den Anti‑Alias ersetzt).

- Randomisierung/Jitter:
  - Training/Fine-Tuning: σ(Blur) und σ(Noise) werden pro Patch gesampelt (robusteres Modell).
  - Visualisierung: ein σ/Noise‑Sample pro Aufruf (kohärente Darstellung über das Volumen), optional per Flags änderbar.

- Wo setze ich welche Flags?
  - `train_ct_sr.py` (Pretraining, Default blurnoise):
    ```bash
    python train_ct_sr.py --data_root "preprocessed_data" --scale 2 --degradation blurnoise --blur_sigma_range 0.7 0.9 --noise_sigma_range_norm 0.001 0.003 --dose_factor_range 0.25 0.5
    ```
  - `finetune_ct_sr.py` (GAN‑Finetuning, Default blurnoise):
    ```bash
    python finetune_ct_sr.py --data_root "preprocessed_data" --scale 2 --degradation blurnoise --blur_sigma_range 0.7 0.9 --noise_sigma_range_norm 0.001 0.003 --dose_factor_range 0.25 0.5
    ```
  - `evaluate_ct_model.py` (metrische Auswertung, Default blurnoise):
    ```bash
    python evaluate_ct_model.py --root "preprocessed_data" --split val --model_path finetune_outputs\best.pth --degradation blurnoise --dose_factor_range 0.25 0.5
    ```
- `evaluate_ct_model.py` (metrische Auswertung, Default blurnoise, also gleich wie oben, nur auf einem Patienten zum Testen, ob alle Metriken korrekt berechnet werden):
    ```bash
    python evaluate_ct_model.py --root preprocessed_data --split val --model_path finetune_outputs\best.pth --max_patients 1
    ```


    Hinweis: Setze hier die gleichen Degradationsparameter wie im Training.
  - `visualize_lr_hr.py` (LR vs HR, Default blurnoise):
    ```bash
    python visualize_lr_hr.py --dicom_folder "preprocessed_data/15041pp" --preset soft_tissue --degradation blurnoise --dose_factor_range 0.25 0.5
    ```
  - `visualize_lr_sr_hr.py` (LR vs SR vs HR, Default blurnoise):
    ```bash
    python visualize_lr_sr_hr.py --dicom_folder "preprocessed_data/15041pp" --model_path rrdb_x2_blurnoise_best.pth --device cuda --preset soft_tissue --degradation blurnoise --dose_factor_range 0.25 0.5
    ```

### Degradations-Sampling und Seed (per-volume / per-slice / deterministisch pro Slice)

- Neue CLI-Optionen (Evaluation/Viewer):
  - `--degradation_sampling {volume,slice,det-slice}` (Default: `volume`)
    - `volume`: Ein Satz Blur/Noise-Parameter für das gesamte Volumen (alle Slices gleich)
    - `slice`: Pro Slice neue Zufallsparameter
    - `det-slice`: Deterministisch je Slice basierend auf `deg_seed + slice_index`
  - `--deg_seed INT` (Default: `42`) – Seed für per-volume bzw. det-slice

- Empfohlene Nutzung zur Reproduzierbarkeit (immer mit `finetune_outputs\best.pth`):

```bash
# Evaluation – per-volume Sampling, reproduzierbar
python evaluate_ct_model.py --root "preprocessed_data" --split val --model_path finetune_outputs\best.pth --device cuda --scale 2 --hu_clip -1000 2000 --degradation blurnoise --noise_sigma_range_norm 0.001 0.003 --dose_factor_range 0.25 0.5 --degradation_sampling volume --deg_seed 42

# Evaluation – deterministisch pro Slice (det-slice)
python evaluate_ct_model.py --root "preprocessed_data" --split val --model_path finetune_outputs\best.pth --device cuda --scale 2 --hu_clip -1000 2000 --degradation blurnoise --noise_sigma_range_norm 0.001 0.003 --dose_factor_range 0.25 0.5 --degradation_sampling det-slice --deg_seed 42

# Evaluation – Sampling-Effekte ausschalten
python evaluate_ct_model.py --root "preprocessed_data" --split val --model_path finetune_outputs\best.pth --device cuda --scale 2 --hu_clip -1000 2000 --degradation clean

# Visualisierung – per-volume Sampling, reproduzierbar
python visualize_lr_sr_hr.py --dicom_folder "preprocessed_data/14797pp" --model_path finetune_outputs\best.pth --device cuda --preset soft_tissue --degradation blurnoise --noise_sigma_range_norm 0.001 0.003 --dose_factor_range 0.25 0.5 --degradation_sampling volume --deg_seed 42

# Visualisierung – deterministisch pro Slice (optional)
python visualize_lr_sr_hr.py --dicom_folder "preprocessed_data/14797pp" --model_path finetune_outputs\best.pth --device cuda --preset soft_tissue --degradation blurnoise --noise_sigma_range_norm 0.001 0.003 --dose_factor_range 0.25 0.5 --degradation_sampling det-slice --deg_seed 42
```

- Hinweis: Die JSON-Summary aus `evaluate_ct_model.py` enthält nun zusätzlich `degradation_sampling` und `deg_seed`.

### Gleiche Metriken: Evaluation und Viewer

- Diese beiden Befehle verwenden dieselbe Metrik-Skalierung.
  - Evaluation: Default ist globalHU [-1000, 2000] (kein Preset/WL/WW gesetzt).
  - Viewer: `--preset global` entspricht exakt HU [-1000, 2000] (WL=500, WW=3000).

```bash
python evaluate_ct_model.py --root preprocessed_data --split test --max_patients 1 --max_slices_per_patient 10 --slice_sampling first
```

```bash
python visualize_lr_sr_hr.py --dicom_folder "preprocessed_data/14655pp" --preset global
```