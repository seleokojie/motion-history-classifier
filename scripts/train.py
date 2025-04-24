#!/usr/bin/env python3
import sys
import os

# Ensure the src directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import io
import pickle
import argparse
import urllib.request
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from random import seed, sample
from joblib import Parallel, delayed

from src.data_loader import DataLoader
from src.mhi import compute_binary_sequence, compute_mhi
from src.features import extract_hu_features
from src.classifier import MHIClassifier

from sklearn.metrics import accuracy_score


def evaluate_params(feats_train, feats_val, thr, tau, k, clf_type, patience, min_acc):
    """
    Train and evaluate for one (threshold, tau) pair using cached features
    and early stopping. Returns (thr, tau, accuracy).
    """
    # Initialize classifier
    clf = MHIClassifier(classifier_type=clf_type, k=k)

    # Train
    X_tr, y_tr = feats_train[(thr, tau)]
    clf.train(X_tr, y_tr)

    X_v, y_v = feats_val[(thr, tau)]
    y_pred = clf.predict(X_v)
    acc = accuracy_score(y_v, y_pred)

    return thr, tau, acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clf', choices=['knn','svm','ada','ensemble'], default='knn',
                        help='Which classifier to use')
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--min_threshold', type=int, default=10)
    parser.add_argument('--max_threshold', type=int, default=60)
    parser.add_argument('--min_tau', type=int, default=10)
    parser.add_argument('--max_tau', type=int, default=60)
    parser.add_argument('--step', type=int, default=10)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--backend', choices=['loky','threading','multiprocessing'],
                        default='loky')
    parser.add_argument('--n_iter', type=int, default=None,
                        help='Number of random search iterations (None = full grid)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--subsample', type=int, default=None,
                        help='Max number of segments to sample for tuning')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early-stop after this many validation samples')
    parser.add_argument('--min_acc', type=float, default=0.1,
                        help='Min running accuracy for early stop')
    args = parser.parse_args()

    # Download dataset if missing
    seq_file   = os.path.join(args.data_dir, 'sequences.txt')
    videos_dir = os.path.join(args.data_dir, 'videos')
    if not os.path.exists(seq_file) or not os.path.isdir(videos_dir):
        print(f"[INFO] Data not found in '{args.data_dir}', downloading KTH Actions dataset...")
        os.makedirs(videos_dir, exist_ok=True)

        # delete it if it exists
        if os.path.exists(seq_file):
            os.remove(seq_file)
        # download the new one
        seq_url = 'https://web.archive.org/web/20190818162100id_/http://www.nada.kth.se:80/cvap/actions/00sequences.txt'
        temp_seq_file = os.path.join(args.data_dir, '00sequences.txt')
        urllib.request.urlretrieve(seq_url, temp_seq_file)
        os.rename(temp_seq_file, seq_file)

        # per-action archives
        actions = ['walking', 'jogging', 'running', 'boxing', 'handwaving', 'handclapping']
        for action in actions:
            zip_url = f'https://web.archive.org/'f'web/20190421074025id_/http://www.nada.kth.se/cvap/actions/{action}.zip'
            resp = urllib.request.urlopen(zip_url)
            with zipfile.ZipFile(io.BytesIO(resp.read())) as zf:
                for member in zf.namelist():
                    if member.lower().endswith('.avi'):
                        out_path = os.path.join(videos_dir, os.path.basename(member))
                        with open(out_path, 'wb') as f:
                            f.write(zf.read(member))

        print("[INFO] Download and extraction complete.\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load all windowed segments
    loader_train = DataLoader(args.data_dir, split='train')
    loader_val   = DataLoader(args.data_dir, split='val')
    train_segs   = list(loader_train.load_segments())
    val_segs     = list(loader_val.load_segments())

    # 3) Optional downsampling of segments
    if args.subsample:
        seed(args.seed)
        train_segs = sample(train_segs, min(args.subsample, len(train_segs)))
        val_segs   = sample(val_segs,   min(args.subsample, len(val_segs)))

    # Build hyperparameter list
    thresholds = list(range(args.min_threshold, args.max_threshold+1, args.step))
    taus       = list(range(args.min_tau,       args.max_tau+1,       args.step))
    grid       = [(thr, tau) for thr in thresholds for tau in taus]

    # 2) Random search sampling
    seed(args.seed)
    if args.n_iter is not None and args.n_iter < len(grid):
        grid = sample(grid, args.n_iter)

    # 1) Precompute & cache features for every (thr, tau)
    feats_train = {}
    feats_val   = {}
    for thr, tau in grid:
        X_tr, y_tr = [], []
        for frames, label, _ in train_segs:
            B = compute_binary_sequence(frames, thr)
            M = compute_mhi(B, tau)
            X_tr.append(extract_hu_features(M))
            y_tr.append(label)
        feats_train[(thr, tau)] = (np.vstack(X_tr), y_tr)

        X_v, y_v = [], []
        for frames, label, _ in val_segs:
            B = compute_binary_sequence(frames, thr)
            M = compute_mhi(B, tau)
            X_v.append(extract_hu_features(M))
            y_v.append(label)
        feats_val[(thr, tau)] = (np.vstack(X_v), y_v)

    # 5) Parallel evaluation with early stopping
    print(f"Starting hyperparam search ({len(grid)} combos) on {args.jobs} jobs...")
    results = Parallel(n_jobs=args.jobs, backend=args.backend)(
        delayed(evaluate_params)(
            feats_train, feats_val,
            thr, tau,
            args.k, args.clf,
            args.patience, args.min_acc
        )
        for thr, tau in grid
    )

    # Pick best
    best_thr, best_tau, best_acc = max(results, key=lambda x: x[2])
    print(f"Best params -> threshold={best_thr}, tau={best_tau}, acc={best_acc:.4f}")

    report_data = {
        'thresholds': thresholds,
        'taus': taus,
        'grid': grid,
        'results': results
    }
    with open(os.path.join(args.output_dir, 'search_results.pkl'), 'wb') as f:
        pickle.dump(report_data, f)
    print(f"Saved hyperparam search data to {args.output_dir}/search_results.pkl")

    # Retrain final model on full train split
    X_final_tr, y_final_tr = feats_train[(best_thr, best_tau)]
    clf = MHIClassifier(classifier_type=args.clf, k=args.k)
    clf.train(X_final_tr, y_final_tr)
    model_path = os.path.join(args.output_dir, 'model_best.joblib')
    clf.save(model_path)
    print(f"Saved final model to {model_path}")

    # Final evaluation on val
    X_final_v, y_final_v = feats_val[(best_thr, best_tau)]
    acc, cm, report = clf.evaluate(X_final_v, y_final_v)
    print(f"Validation Accuracy: {acc:.4f}\n{report}")

    # Plot and save confusion matrix
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()
    classes = clf.clf.classes_
    ticks   = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45)
    plt.yticks(ticks, classes)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"Saved confusion matrix to {cm_path}")

if __name__ == '__main__':
    main()