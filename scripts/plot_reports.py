#!/usr/bin/env python3
import sys
import os

# Ensure the src directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, colormaps
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, auc
from sklearn.preprocessing import label_binarize

from src.data_loader import DataLoader
from src.mhi import compute_binary_sequence, compute_mhi
from src.features import extract_hu_features
from src.classifier import MHIClassifier


def plot_confusion_matrix(y_true, y_pred, classes, output_path):
    """
    Plots a confusion matrix where the color scale is percentage (0–100%),
    but each cell is annotated with both the raw count and the percentage.

    Parameters:
    - y_true: list or array of ground-truth labels
    - y_pred: list or array of predicted labels
    - classes: list of class names, in the order to index the matrix
    - output_path: filepath to save the resulting figure (e.g. 'cm.png')
    """
    # Compute raw conf. matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    # Normalize by true-class totals → fractions in [0,1]
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, None]

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm_norm * 100,
                   interpolation='nearest',
                   cmap= colormaps['YlGnBu'],
                   norm=colors.Normalize(vmin=0, vmax=100))

    # Colorbar labeled as percentage
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Percentage (%)', rotation=270, labelpad=15)

    # Axis ticks/labels
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticklabels(classes)

    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion Matrix\n(counts + %)')

    # Annotate each cell with count and percent
    thresh = cm_norm.max() * 100 / 2.0  # half of max percentage
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100
            text = f"{count}\n{pct:.1f}%"
            ax.text(j, i, text,
                    ha="center", va="center",
                    color="white" if (cm_norm[i, j] * 100) > thresh else "black")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_prf_bars(y_true, y_pred, classes, output_path):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=classes)
    x = np.arange(len(classes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(x - width, p, width, label='Precision')
    ax.bar(x       , r, width, label='Recall')
    ax.bar(x + width, f1, width, label='F1-score')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_ylabel('Score')
    ax.set_title('Per-class Precision / Recall / F1')
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_multiclass_roc(clf, X, y, classes, output_path):
    Y = label_binarize(y, classes=classes)
    y_score = clf.predict_proba(X)
    fig, ax = plt.subplots(figsize=(8,6))
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(Y[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f'{cls} (AUC={roc_auc:.2f})')
    ax.plot([0,1],[0,1],'k--', lw=1)
    ax.set_xlim([0.0,1.0]); ax.set_ylim([0.0,1.05])
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('One-vs-Rest ROC Curves')
    ax.legend(loc='lower right', fontsize='small')
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def show_mhi_example(frames, thr, tau, prefix, output_dir):
    binaries = compute_binary_sequence(frames, thr)
    mhi      = compute_mhi(binaries, tau)
    fig, axes = plt.subplots(1,3,figsize=(12,4))
    axes[0].imshow(frames[len(frames)//2],cmap='gray'); axes[0].set_title('Frame')
    axes[1].imshow(binaries[len(binaries)//2],cmap='gray'); axes[1].set_title('Binary')
    axes[2].imshow(mhi, cmap='gray'); axes[2].set_title('MHI')
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    path = os.path.join(output_dir, f"{prefix}_mhi_pipeline.png")
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--results', required=True,
                        help='Pickle file with grid & results list')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--split', default='val')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load search results (with full thresholds & taus)

    with open(args.results, 'rb') as f:
        data = pickle.load(f)
    thresholds = data['thresholds']  # e.g. [10,20,30,40,50,60]
    taus = data['taus']  # e.g. [10,20,30,40,50,60]
    results = data['results']  # list of (thr, tau, accuracy)
    acc_dict = {(thr, tau): acc for thr, tau, acc in results}

    # 2) Load model
    clf = MHIClassifier.load(args.model_path)
    classes = list(clf.clf.classes_)

    # 3) Recompute validation set features
    loader = DataLoader(args.data_dir, split=args.split)
    val_data = list(loader.load_segments())
    # Use best params:
    best_thr, best_tau, _ = max(results, key=lambda x: x[2])
    X_val, y_val = [], []
    for frames, label, _ in val_data:
        B = compute_binary_sequence(frames, best_thr)
        M = compute_mhi(B, best_tau)
        X_val.append(extract_hu_features(M))
        y_val.append(label)
    X_val = np.vstack(X_val)
    y_pred = clf.predict(X_val)

    # 4) Generate plots
    plot_confusion_matrix(y_val, y_pred, classes,
                          os.path.join(args.output_dir,'confusion_matrix_report.png'))
    plot_prf_bars(y_val, y_pred, classes,
                  os.path.join(args.output_dir,'prf_bars.png'))
    plot_multiclass_roc(clf.clf, X_val, y_val, classes,
                        os.path.join(args.output_dir,'roc_multiclass.png'))

    # 5) Example MHIs (pick one correct & one incorrect)
    correct = [(f,l,idx) for idx,(f,l,_) in enumerate(val_data) if l==y_pred[idx]]
    incorrect = [(f,l,idx) for idx,(f,l,_) in enumerate(val_data) if l!=y_pred[idx]]
    if correct:
        show_mhi_example(correct[0][0], best_thr, best_tau, 'success', args.output_dir)
    if incorrect:
        show_mhi_example(incorrect[0][0], best_thr, best_tau, 'failure', args.output_dir)

    print(f"All report plots saved under {args.output_dir}")

if __name__ == '__main__':
    main()
