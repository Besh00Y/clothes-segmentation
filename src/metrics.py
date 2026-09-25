import torch
from torchmetrics.classification import (
    BinaryJaccardIndex,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)


def get_metrics(device):
    metrics = {
        "iou": BinaryJaccardIndex().to(device),
        "dice": BinaryF1Score().to(device),
        "precision": BinaryPrecision().to(device),
        "recall": BinaryRecall().to(device),
    }

    return metrics


def update_metrics(metrics, logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).int()
    targets = targets.int()

    metrics["iou"].update(preds, targets)
    metrics["dice"].update(preds, targets)
    metrics["precision"].update(preds, targets)
    metrics["recall"].update(preds, targets)


def compute_metrics(metrics):
    return {
        "iou": metrics["iou"].compute().item(),
        "dice": metrics["dice"].compute().item(),
        "precision": metrics["precision"].compute().item(),
        "recall": metrics["recall"].compute().item(),
    }


def reset_metrics(metrics):
    for metric in metrics.values():
        metric.reset()