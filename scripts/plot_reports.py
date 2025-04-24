#!/usr/bin/env python3
import sys
import os
import json
import argparse

# Ensure the src directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, auc
from sklearn.preprocessing import label_binarize

from src.data_loader import DataLoader
from src.mhi import compute_binary_sequence, compute_mhi
from src.features import extract_hu_features
from src.classifier import MHIClassifier


def plot_confusion_matrix(y_true, y_pred, classes, output_path, title_suffix=''):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, None]
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm_norm * 100, interpolation='nearest', cmap='YlGnBu', vmin=0, vmax=100)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Percentage (%)', rotation=270, labelpad=15)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticklabels(classes)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    title = 'Confusion Matrix'
    if title_suffix:
        title += f' ({title_suffix})'
    ax.set_title(title)
    thresh_pct = cm_norm.max() * 100 / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100
            text = f"{count}\n{pct:.1f}%"
            ax.text(j, i, text, ha='center', va='center',
                    color='white' if pct > thresh_pct else 'black')
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_prf_bars(y_true, y_pred, classes, output_path, title_suffix=''):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=classes)
    x = np.arange(len(classes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, p, width, label='Precision')
    ax.bar(x      , r, width, label='Recall')
    ax.bar(x + width, f1, width, label='F1-score')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_ylabel('Score')
    title = 'Per-class Precision / Recall / F1'
    if title_suffix:
        title += f' ({title_suffix})'
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_multiclass_roc(clf, X, y, classes, output_path, title_suffix=''):
    Y_bin = label_binarize(y, classes=classes)
    y_score = clf.predict_proba(X)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(Y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f'{cls} (AUC={roc_auc:.2f})')
    ax.plot([0,1], [0,1], 'k--', lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    title = 'One-vs-Rest ROC Curves'
    if title_suffix:
        title += f' ({title_suffix})'
    ax.set_title(title)
    ax.legend(loc='lower right', fontsize='small')
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def show_mhi_example(frames, thr, tau, prefix, output_dir):
    B = compute_binary_sequence(frames, threshold=thr)
    M = compute_mhi(B, tau)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    idx = len(frames) // 2
    axes[0].imshow(frames[idx], cmap='gray')
    axes[0].set_title('Frame')
    axes[1].imshow(B[idx-1], cmap='gray')
    axes[1].set_title('Binary')
    axes[2].imshow(M, cmap='gray')
    axes[2].set_title(f'MHI (thr={thr})')
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    path = os.path.join(output_dir, f"{prefix}_mhi_thr{thr}.png")
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',    default='data',
                        help='Path to data directory (default: data)')
    parser.add_argument('--model_path',  default='outputs/model_best.joblib',
                        help='Path to trained model (default: outputs/model_best.joblib)')
    parser.add_argument('--results',     dest='thresholds_file',
                        default='outputs/thresholds_per_action.json',
                        help='JSON file for per-action thresholds (default: outputs/thresholds_per_action.json)')
    parser.add_argument('--threshold',   type=int, default=None,
                        help='Override global threshold for recomputing features')
    parser.add_argument('--tau',         type=int, default=260,
                        help='History length τ (default: 260)')
    parser.add_argument('--output_dir',  default='outputs/reports',
                        help='Directory to save report plots (default: outputs/reports)')
    parser.add_argument('--split',       default='val',
                        help='Data split to use (default: val)')
    args = parser.parse_args()

    # load thresholds mapping
    if not os.path.exists(args.thresholds_file):
        print(f"[ERROR] Thresholds file not found: {args.thresholds_file}")
        sys.exit(1)
    with open(args.thresholds_file) as f:
        per_act = json.load(f)

    # determine global threshold
    if args.threshold is not None:
        global_thr = args.threshold
    elif 'global' in per_act:
        global_thr = per_act['global']
    else:
        vals = list(per_act.values())
        global_thr = int(np.median(vals))
        print(f"[WARNING] 'global' key missing; using median per-action threshold={global_thr}")

    print(f"[INFO] Using threshold={global_thr}, τ={args.tau}\n")

    # load model
    if not os.path.exists(args.model_path):
        print(f"[ERROR] Model file not found: {args.model_path}")
        sys.exit(1)
    clf = MHIClassifier.load(args.model_path)
    classes = list(clf.clf.classes_)

    # load validation data
    loader = DataLoader(args.data_dir, split=args.split)
    val_data = list(loader.load_segments())

    # compute features & predictions with global threshold
    X, y_true = [], []
    for frames, label, _ in val_data:
        B = compute_binary_sequence(frames, threshold=global_thr)
        M = compute_mhi(B, tau=args.tau)
        X.append(extract_hu_features(M))
        y_true.append(label)
    X = np.vstack(X)
    y_pred = clf.predict(X)

    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # generate global plots
    plot_confusion_matrix(y_true, y_pred, classes,
                          os.path.join(args.output_dir, 'confusion_matrix_global.png'), 'global')
    plot_prf_bars(y_true, y_pred, classes,
                  os.path.join(args.output_dir, 'prf_bars_global.png'), 'global')
    plot_multiclass_roc(clf.clf, X, y_true, classes,
                        os.path.join(args.output_dir, 'roc_multiclass_global.png'), 'global')

    # generate per-action recall bar chart
    recalls = []
    for act in classes:
        mask = [i for i,(_,label,_) in enumerate(val_data) if label == act]
        y_true_act = [y_true[i] for i in mask]
        y_pred_act = [y_pred[i] for i in mask]
        if mask:
            _, recall, _, _ = precision_recall_fscore_support(
                y_true_act, y_pred_act, labels=[act], zero_division=0
            )
            recalls.append(recall[0])
        else:
            recalls.append(0.0)
    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(classes, recalls)
    ax.set_ylabel('Recall')
    ax.set_xlabel('Action')
    ax.set_title('Per-action Recall (global threshold)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, 'per_action_recall_global.png'), dpi=150)
    plt.close(fig)

    # example MHIs with per-action thresholds and annotated names
    success_idxs = [i for i,(f,l,_) in enumerate(val_data) if l == y_pred[i]]
    failure_idxs = [i for i,(f,l,_) in enumerate(val_data) if l != y_pred[i]]
    for prefix, idx_list in [('success', success_idxs), ('failure', failure_idxs)]:
        if idx_list:
            idx = idx_list[0]
            frames, label, _ = val_data[idx]
            thr_act = per_act.get(label, global_thr)
            show_mhi_example(frames, thr_act, args.tau, f"{prefix}_{label}", args.output_dir)

    print(f"All report plots saved under {args.output_dir}")

if __name__ == '__main__':
    main()