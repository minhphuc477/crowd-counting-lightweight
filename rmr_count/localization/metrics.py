from __future__ import annotations

from typing import Any
import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


def match_points(
    preds: np.ndarray | list[list[float]],
    gts: np.ndarray | list[list[float]],
    sigma: float,
) -> tuple[int, int, int]:
    """Matches predicted points to ground-truth points under distance threshold sigma.

    Uses spatial KDTree candidate pruning and sparse Hopcroft-Karp maximum bipartite
    matching. This uses O(M) memory rather than allocating dense (M, N) distance matrices,
    providing complete immunity against memory allocation errors on dense crowd images.

    Args:
        preds: Array-like of predicted coordinates, shape (M, 2) [x, y].
        gts: Array-like of ground-truth coordinates, shape (N, 2) [x, y].
        sigma: Maximum allowable Euclidean distance in pixels to count as a match.

    Returns:
        tuple (tp, fp, fn):
            tp: Count of true positive matched pairs with distance <= sigma.
            fp: Count of false positives (unmatched predictions: M - tp).
            fn: Count of false negatives (unmatched ground truths: N - tp).
    """
    preds_arr = np.asarray(preds, dtype=np.float32)
    gts_arr = np.asarray(gts, dtype=np.float32)

    m = len(preds_arr) if preds_arr.ndim == 2 and preds_arr.shape[1] == 2 else 0
    n = len(gts_arr) if gts_arr.ndim == 2 and gts_arr.shape[1] == 2 else 0

    if m == 0 and n == 0:
        return 0, 0, 0
    if m == 0:
        return 0, 0, n
    if n == 0:
        return 0, m, 0

    tree = KDTree(gts_arr)
    neighbors = tree.query_ball_point(preds_arr, r=float(sigma))

    rows: list[int] = []
    cols: list[int] = []
    for i, nbrs in enumerate(neighbors):
        for j in nbrs:
            rows.append(i)
            cols.append(j)

    if not rows:
        return 0, m, n

    data = np.ones(len(rows), dtype=bool)
    adj = csr_matrix((data, (rows, cols)), shape=(m, n))
    matching = maximum_bipartite_matching(adj, perm_type="column")

    tp = int(np.sum(matching >= 0))
    fp = m - tp
    fn = n - tp

    return tp, fp, fn


class LocalizationMeter:
    """Accumulates point localization evaluation statistics across multiple distance thresholds."""

    def __init__(self, sigmas: list[float] | tuple[float, ...] = (4.0, 8.0)) -> None:
        self.sigmas = [float(s) for s in sigmas]
        self.reset()

    def reset(self) -> None:
        self.total_preds = 0
        self.total_gts = 0
        self.tp: dict[float, int] = {s: 0 for s in self.sigmas}
        self.fp: dict[float, int] = {s: 0 for s in self.sigmas}
        self.fn: dict[float, int] = {s: 0 for s in self.sigmas}
        self.image_metrics: dict[float, list[dict[str, float]]] = {s: [] for s in self.sigmas}

    def update(
        self,
        preds: np.ndarray | list[list[float]],
        gts: np.ndarray | list[list[float]],
    ) -> dict[float, tuple[int, int, int]]:
        preds_arr = np.asarray(preds, dtype=np.float32)
        gts_arr = np.asarray(gts, dtype=np.float32)

        m = len(preds_arr) if preds_arr.ndim == 2 and preds_arr.shape[1] == 2 else 0
        n = len(gts_arr) if gts_arr.ndim == 2 and gts_arr.shape[1] == 2 else 0

        self.total_preds += m
        self.total_gts += n

        results = {}
        for s in self.sigmas:
            tp, fp, fn = match_points(preds_arr, gts_arr, sigma=s)
            self.tp[s] += tp
            self.fp[s] += fp
            self.fn[s] += fn

            # Per-image macro tracking
            p = float(tp / m) if m > 0 else (1.0 if n == 0 else 0.0)
            r = float(tp / n) if n > 0 else (1.0 if m == 0 else 0.0)
            f1 = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            self.image_metrics[s].append({"precision": p, "recall": r, "f1": f1})
            results[s] = (tp, fp, fn)

        return results

    def compute_summary(self) -> dict[str, Any]:
        """Computes micro and macro precision, recall, and F1 per distance threshold."""
        summary: dict[str, Any] = {
            "total_predictions": self.total_preds,
            "total_ground_truth": self.total_gts,
            "thresholds": {},
        }

        for s in self.sigmas:
            tp = self.tp[s]
            fp = self.fp[s]
            fn = self.fn[s]

            # Global micro metrics
            micro_p = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            micro_r = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            micro_f1 = float(2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0

            # Macro metrics (mean of per-image scores)
            img_scores = self.image_metrics[s]
            if img_scores:
                macro_p = float(np.mean([x["precision"] for x in img_scores]))
                macro_r = float(np.mean([x["recall"] for x in img_scores]))
                macro_f1 = float(np.mean([x["f1"] for x in img_scores]))
            else:
                macro_p = macro_r = macro_f1 = 0.0

            key = f"sigma_{int(s)}" if s.is_integer() else f"sigma_{s:.1f}"
            summary["thresholds"][key] = {
                "sigma": s,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "micro_precision": micro_p,
                "micro_recall": micro_r,
                "micro_f1": micro_f1,
                "macro_precision": macro_p,
                "macro_recall": macro_r,
                "macro_f1": macro_f1,
            }

        return summary
