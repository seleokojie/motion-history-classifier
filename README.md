# MHI Activity Classifier

This project implements human activity recognition (walking, jogging, running, boxing, waving, clapping) from video using **Motion History Images** (MHIs) and handcrafted Hu moments. You get:

- MHI & Hu‐moment feature extraction (no pre-built MHI or Hu functions).
- K-Nearest-Neighbors, SVM, AdaBoost or a soft-voting ensemble (KNN+SVM+RF+AdaBoost).
- Grid-search / random-search over threshold & history length, with parallelism.
- Feature-caching, down-sampling, and early-stopping to speed up tuning.
- Real-time annotation script for new videos.

---

## Repository Layout
```
project_root/
├── data/
│   ├── sequences.txt
│   └── videos/           # all .avi clips here
├── src/
│   ├── data_loader.py    # parse sequences.txt & load frame segments
│   ├── mhi.py            # compute frame-diff binaries & MHIs
│   ├── features.py       # compute central & scale-invariant moments
│   ├── classifier.py     # KNN, SVM, AdaBoost & VotingClassifier wrapper
│   └── utils.py          # e.g. frame annotation
├── scripts/
│   ├── train.py          # hyperparam tuning & final training script
│   └── annotate.py       # real-time video annotation using trained model
├── outputs/
│   ├── model_best.joblib
│   ├── confusion_matrix.png
│   └── annotated_videos/
├── requirements.txt
└── README.md
```
---

## Setup

1. Clone this repo:
  ```bash
   git clone https://github.com/seleokojie/mhi-activity-classifier.git
   cd mhi-activity-classifier
  ```

2. Create & activate a Python 3.8+ virtual environment:
  ```bash
   python3 -m venv venv
   source venv/bin/activate        # macOS/Linux or
   venv\Scripts\activate.bat       # Windows
  ```

3. Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

4. Ensure `src/` is on your Python path (if needed):
  ```bash
   export PYTHONPATH="$PWD"        # macOS/Linux or
   set PYTHONPATH=%CD%`            # Windows PowerShell
  ```

---

## Usage

### Training & Hyperparameter Search

Quick test run (accuracy doesn’t matter):
```bash
python scripts/train.py \\
  --data_dir data \\
  --output_dir outputs \\
  --clf knn \\
  --k 1 \\
  --min_threshold 30 --max_threshold 30 --step 1 \\
  --min_tau 30       --max_tau 30       --step 1 \\
  --jobs 1 --backend threading \\
  --n_iter 1 \\
  --subsample 5 \\
  --patience 1 --min_acc 0.5
```

Parallel random search with soft-voting ensemble:
```bash
python scripts/train.py \\
  --data_dir data \\
  --output_dir outputs \\
  --clf ensemble \\
  --k 5 \\
  --min_threshold 10 --max_threshold 60 --step 10 \\
  --min_tau 10       --max_tau 60       --step 10 \\
  --jobs 8 --backend threading \\
  --n_iter 20 --seed 42 \\
  --subsample 150 \\
  --patience 10 --min_acc 0.1
```

---

## Annotate a New Video

```bash
python scripts/annotate.py \\
  --model_path outputs/model_best.joblib \\
  --input_video data/videos/person22_walking_d1.avi \\
  --output_video outputs/annotated_videos/walk22.avi \\
  --threshold 30 \\
  --tau 30
```
```bash
python scripts/annotate.py \\
  --model_path outputs/model_best.joblib \\
  --input_video data/videos/person22_walking_d1.avi \\
  --output_video outputs/annotated_videos/walk22.avi \\
  --threshold 30 \\
  --tau 30
```

```bash
python scripts/annotate.py \\
  --model_path outputs/model_best.joblib \\
  --input_video data/videos/person22_walking_d1.avi \\
  --output_video outputs/annotated_videos/walk22.avi \\
  --threshold 30 \\
  --tau 30
```

---

## Speed-Up Tips

- Feature Caching: Precompute MHI+Hu features once (built into `train.py`)
- Random Search: Use `--n_iter` to avoid full grid
- Down-sampling: `--subsample` picks a small subset of segments
- Early-Stopping: `--patience` & `--min_acc` bail on bad hyperparams
- Parallelism: `--jobs` + `--backend` threading avoids Windows pickling