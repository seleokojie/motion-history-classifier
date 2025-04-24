#!/usr/bin/env python3
import sys
import os
import io
import json
import argparse
import urllib.request
import zipfile
from random import seed, sample
from joblib import Parallel, delayed

# Ensure the src directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import DataLoader
from src.mhi import compute_binary_sequence, compute_mhi
from src.features import extract_hu_features
from src.classifier import MHIClassifier


def evaluate_threshold(thr, feats_train, feats_val, clf_type, k):
    """
    Train on train‐split with threshold=thr, tau fixed.
    Return (thr, y_val, y_pred, classes).
    """
    X_tr, y_tr = feats_train[thr]
    clf = MHIClassifier(classifier_type=clf_type, k=k)
    clf.train(X_tr, y_tr)

    X_v, y_v = feats_val[thr]
    y_pred = clf.predict(X_v)
    return thr, y_v, y_pred, list(clf.clf.classes_)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clf', choices=['knn','svm','ada','ensemble'],
                        default='knn', help='Which classifier to use')
    parser.add_argument('--data_dir',    required=True)
    parser.add_argument('--output_dir',  required=True)
    parser.add_argument('--min_threshold', type=int, default=10,
                        help='Smallest threshold to try')
    parser.add_argument('--max_threshold', type=int, default=60,
                        help='Largest threshold to try')
    parser.add_argument('--step', type=int, default=10,
                        help='Step size for threshold grid')
    parser.add_argument('--tau', type=int, default=260,
                        help='Fixed history length τ for MHI (default:260)')
    parser.add_argument('--k', type=int, default=5,
                        help='Number of neighbors for KNN or ensemble')
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--backend', choices=['loky','threading','multiprocessing'],
                        default='loky')
    parser.add_argument('--n_iter', type=int, default=None,
                        help='If set, sample this many thresholds at random')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--subsample', type=int, default=None,
                        help='Max number of segments to sample for tuning')
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

    # 2) Load segments
    loader_train = DataLoader(args.data_dir, split='train')
    loader_val = DataLoader(args.data_dir, split='val')
    train_segs = list(loader_train.load_segments())
    val_segs = list(loader_val.load_segments())

    # 3) Optional downsampling
    seed(args.seed)
    if args.subsample:
        train_segs = sample(train_segs, min(args.subsample, len(train_segs)))
        val_segs = sample(val_segs, min(args.subsample, len(val_segs)))

    # 4) Build threshold grid
    thresholds = list(range(args.min_threshold,
                            args.max_threshold + 1,
                            args.step))
    seed(args.seed)
    if args.n_iter is not None and args.n_iter < len(thresholds):
        thresholds = sample(thresholds, args.n_iter)
    print(f"[INFO] Tuning thresholds={thresholds} with τ={args.tau}\n")

    # 5) Precompute & cache features for each threshold (τ fixed)
    feats_train = {}
    feats_val = {}
    for thr in thresholds:
        X_tr, y_tr = [], []
        for frames, label, _ in train_segs:
            B = compute_binary_sequence(frames, threshold=thr)
            M = compute_mhi(B, tau=args.tau)
            X_tr.append(extract_hu_features(M))
            y_tr.append(label)
        feats_train[thr] = (np.vstack(X_tr), y_tr)

        X_v, y_v = [], []
        for frames, label, _ in val_segs:
            B = compute_binary_sequence(frames, threshold=thr)
            M = compute_mhi(B, tau=args.tau)
            X_v.append(extract_hu_features(M))
            y_v.append(label)
        feats_val[thr] = (np.vstack(X_v), y_v)

    # 6) Tune: get y_val & y_pred for each threshold in parallel
    tasks = Parallel(n_jobs=args.jobs, backend=args.backend)(
        delayed(evaluate_threshold)(
            thr, feats_train, feats_val, args.clf, args.k
        ) for thr in thresholds
    )

    # 7) Compute the best threshold *per action* by recall
    #   gather all action labels
    all_actions = sorted({label for _, label_list, _ in val_segs for label in [label_list]})
    best_thresh = {act: None for act in all_actions}
    best_score = {act: -1.0 for act in all_actions}

    for thr, y_val, y_pred, classes in tasks:
        y_val_arr = np.array(y_val)
        y_pred_arr = np.array(y_pred)
        for act in all_actions:
            mask = (y_val_arr == act)
            if mask.sum() == 0:
                continue
            recall = (y_pred_arr[mask] == act).mean()
            if recall > best_score[act]:
                best_score[act] = recall
                best_thresh[act] = thr

    # 8) Report & save
    print("[INFO] Best thresholds per action (by recall):")
    for act in all_actions:
        print(f"   {act:12s} → thr={best_thresh[act]}  (recall={best_score[act]:.3f})")

    with open(os.path.join(args.output_dir, 'thresholds_per_action.json'), 'w') as f:
        json.dump(best_thresh, f, indent=2)
    print(f"\n[INFO] Saved per-action thresholds to {args.output_dir}/thresholds_per_action.json")

    # 9) (Optional) Re-train final model on full train split
    #    using *global* threshold that maximizes overall val accuracy
    agg = []
    for thr, y_val, y_pred, _ in tasks:
        agg.append((thr, (np.array(y_val) == np.array(y_pred)).mean()))
    best_global_thr, best_acc = max(agg, key=lambda x: x[1])
    print(f"\n[INFO] Best global threshold={best_global_thr} → val-acc={best_acc:.4f}")
    X_tr_final, y_tr_final = feats_train[best_global_thr]
    clf = MHIClassifier(classifier_type=args.clf, k=args.k)
    clf.train(X_tr_final, y_tr_final)
    model_path = os.path.join(args.output_dir, 'model_best.joblib')
    clf.save(model_path)
    print(f"[INFO] Saved final model (τ={args.tau}, thr={best_global_thr}) to {model_path}")

    # 10) Final evaluation on val set
    X_v_final, y_v_final = feats_val[best_global_thr]
    acc, cm, report = clf.evaluate(X_v_final, y_v_final)
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