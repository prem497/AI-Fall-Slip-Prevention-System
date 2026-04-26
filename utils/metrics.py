"""
Evaluation metrics for the Fall & Slip Prevention System.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    average_precision_score, roc_curve, precision_recall_curve
)


def compute_metrics(y_true, y_pred, y_prob=None, class_names=None):
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities (optional, for AUC).
        class_names: List of class names.
    Returns:
        dict: Dictionary of metrics.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) == 2:
                prob = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
                metrics['auc_roc'] = roc_auc_score(y_true, prob)
                metrics['avg_precision'] = average_precision_score(y_true, prob)
            else:
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
        except Exception:
            metrics['auc_roc'] = None
    if class_names:
        metrics['classification_report'] = classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0
        )
    return metrics


def print_metrics(metrics):
    """Pretty print evaluation metrics."""
    print(f"\n{'=' * 50}")
    print(f"  Evaluation Results")
    print(f"{'=' * 50}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    if metrics.get('auc_roc') is not None:
        print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    if metrics.get('avg_precision') is not None:
        print(f"  Avg Prec:  {metrics['avg_precision']:.4f}")
    print(f"\n  Confusion Matrix:")
    for row in metrics['confusion_matrix']:
        print(f"    {row}")
    if 'classification_report' in metrics:
        print(f"\n{metrics['classification_report']}")
    print(f"{'=' * 50}")


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self, name=''):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"{self.name}: {self.avg:.4f}"


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience=10, min_delta=1e-4, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        improved = (score < self.best_score - self.min_delta) if self.mode == 'min' \
            else (score > self.best_score + self.min_delta)
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        return False
