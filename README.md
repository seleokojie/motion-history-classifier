# MHI Activity Classifier

This project implements human activity recognition (walking, jogging, running, boxing, waving, clapping) from video using **Motion History Images** (MHIs) and handcrafted Hu moments. The pipeline covers data loading, feature extraction, classifier training, model evaluation, and real-time annotation.

---

## Repository Layout
```
project_root/
├── data/
│   ├── sequences.txt
│   └── videos/           # all .avi clips here
├── src/                  # core modules
│   ├── data_loader.py    # parse sequences.txt & load/cached frame segments
│   ├── mhi.py            # compute frame-diff binaries & MHIs (CPU/GPU)
│   ├── features.py       # compute scale-invariant central (Hu) moments
│   ├── classifier.py     # KNN, SVM, AdaBoost, RandomForest & soft-voting ensemble
│   └── utils.py          # frame annotation helper
├── scripts/
│   ├── train.py          # hyperparam search & model training
│   ├── plot_reports.py   # performance diagnostics & visualizations
│   └── annotate.py       # real-time video annotation
├── outputs/
│   ├── model_best.joblib
│   ├── thresholds_per_action.json
│   ├── annotated_videos/
│   └── reports/          # report plots (confusion matrix, ROC, etc.)
├── requirements.txt
└── README.md
```

---

## Setup

1. **Clone** the repo and enter the directory:
   ```bash
   git clone https://github.com/seleokojie/mhi-activity-classifier.git
   cd mhi-activity-classifier
   ```

2. **Create & activate** a Python 3.8+ virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate        # macOS/Linux
   venv\Scripts\activate.bat       # Windows PowerShell
   ```

3. **Install** dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional)** Ensure `src/` is on your Python path:
   ```bash
   export PYTHONPATH="$PWD"         # macOS/Linux
   set PYTHONPATH=%CD%              # Windows PowerShell
   ```

---

## Usage

### 1. Training & Hyperparameter Search (`scripts/train.py`)

Runs grid-search or randomized search over motion-threshold values to find per-action and global thresholds, then trains a final classifier.

```bash
python scripts/train.py \
  --data_dir data \
  --output_dir outputs \
  --clf ensemble \
  --k 5 \
  --min_threshold 10 --max_threshold 400 --step 10 \
  --tau 260 \
  --jobs 8 --backend loky \
  --n_iter 25 --seed 42 \
  --subsample 200
```

**Flags**:
- `--data_dir`: Path to data directory containing `sequences.txt` and `videos/`.
- `--output_dir`: Directory where `model_best.joblib` and `thresholds_per_action.json` will be saved.
- `--clf`: Classifier type (`knn`, `svm`, `ada`, or `ensemble`).
- `--k`: Number of neighbors for KNN (and ensemble KNN component).
- `--min_threshold`, `--max_threshold`, `--step`: Range and step size for motion-difference threshold grid.
- `--tau`: Motion History Image decay parameter (history length).
- `--jobs`, `--backend`: Parallelism settings for hyperparameter tuning.
- `--n_iter`: If set, randomly samples this many thresholds instead of full grid.
- `--seed`: Random seed for reproducibility.
- `--subsample`: Maximum number of segments to uniformly sample for tuning (speeds up early iterations).

Upon completion, you’ll see printed per-action best thresholds and a final global threshold. Outputs:
- `outputs/thresholds_per_action.json`
- `outputs/model_best.joblib`

---

### 2. Performance Reporting (`scripts/plot_reports.py`)

Generates diagnostic plots on a held-out split (default `val`).

```bash
python scripts/plot_reports.py --threshold 10
```

**Flags**:
- `--threshold`: Override global threshold for feature recomputation (default: median of per-action thresholds).
- `--tau`: History length τ for MHIs (default: 260).
- `--model_path`: Path to trained model (default: `outputs/model_best.joblib`).
- `--data_dir`: Dataset root (default: `data`).
- `--output_dir`: Directory to save plots (default: `outputs/reports`).
- `--split`: Data split to visualize (`train`, `val`, or `test`).

Plots generated include:
- Normalized confusion matrix (`confusion_matrix_global.png`)
- Precision/Recall/F1 bar chart (`prf_bars_global.png`)
- One-vs-Rest ROC curves (`roc_multiclass_global.png`)
- Per-action recall bar chart (`per_action_recall_global.png`)
- Example MHIs for successes and failures

---

### 3. Real-Time Annotation (`scripts/annotate.py`)

Annotates each frame of a video with the predicted action label.

```bash
python scripts/annotate.py \
  --model_path outputs/model_best.joblib \
  --person 22 \
  --action handclapping \
  --condition d1
```

**Flags**:
- `--model_path`: Path to the trained classifier (default: `outputs/model_best.joblib`).
- `--person`: Person ID matching your `data/videos/person{ID}_{action}_{cond}_uncomp.avi` naming.
- `--action`: Action (`walking`, `jogging`, `running`, `boxing`, `handwaving`, `handclapping`).
- `--condition`: Condition label (e.g., `d1`–`d4`).
- `--threshold`: Optional override of per-action threshold (default: loaded from `thresholds_per_action.json`).
- `--tau`: History length τ (default: 260).

The script reads `data/videos/person{ID}_{action}_{cond}_uncomp.avi`, applies a sliding window of τ frames to compute MHIs, extracts Hu moments, predicts labels, and writes an annotated `.avi` to `outputs/annotated_videos/`.

---