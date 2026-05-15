import numpy as np
import pandas as pd
from sklearn import metrics


def roc_auc(y_true: np.ndarray, pred: np.ndarray) -> tuple[pd.DataFrame, float]:
    fpr, tpr, threshold = metrics.roc_curve(y_true, pred)
    curve = pd.DataFrame({"roc_fpr": fpr, "roc_tpr": tpr, "roc_threshold": threshold})
    return curve, metrics.auc(fpr, tpr)


def pr_auc(y_true: np.ndarray, pred: np.ndarray) -> tuple[pd.DataFrame, float]:
    precision, recall, threshold = metrics.precision_recall_curve(y_true, pred)
    curve = pd.DataFrame(
        {
            "pr_precision": precision,
            "pr_recall": recall,
            "pr_threshold": np.append(threshold, np.nan),
        }
    )
    return curve, metrics.auc(recall, precision)
