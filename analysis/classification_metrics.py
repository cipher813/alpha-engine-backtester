"""
classification_metrics.py — Precision, recall, F1, and confusion matrix utilities.

Used by analysis modules to compute standard classification metrics at each
decision boundary in the pipeline.
"""


def compute_binary_metrics(
    tp: int, fp: int, fn: int, tn: int = 0,
) -> dict:
    """Compute precision, recall, F1, and accuracy from a confusion matrix.

    Args:
        tp: True positives (correctly selected winners)
        fp: False positives (incorrectly selected losers)
        fn: False negatives (missed winners)
        tn: True negatives (correctly excluded losers)

    Returns:
        Dict with precision, recall, f1, accuracy, and counts.
        All rates are floats in [0, 1] or None if undefined.
    """
    total = tp + fp + fn + tn

    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None

    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None

    accuracy = (tp + tn) / total if total > 0 else None

    return {
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
        "accuracy": round(accuracy, 4) if accuracy is not None else None,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n": total,
    }


def compute_from_boolean_arrays(selected: list[bool], positive: list[bool]) -> dict:
    """Compute metrics from two parallel boolean arrays.

    Args:
        selected: Whether the system selected/predicted this item (e.g., passed filter, picked by team).
        positive: Whether the item had a positive outcome (e.g., beat SPY).

    Both arrays must have the same length.
    """
    if len(selected) != len(positive):
        raise ValueError(f"Array length mismatch: {len(selected)} vs {len(positive)}")

    tp = fp = fn = tn = 0
    for s, p in zip(selected, positive):
        if s and p:
            tp += 1
        elif s and not p:
            fp += 1
        elif not s and p:
            fn += 1
        else:
            tn += 1

    return compute_binary_metrics(tp, fp, fn, tn)
