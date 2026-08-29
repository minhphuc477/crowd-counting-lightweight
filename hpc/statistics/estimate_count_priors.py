from __future__ import annotations

import numpy as np


def estimate_statistics(image_counts, positive_y16_counts):
    image_counts = np.asarray(image_counts, dtype=np.float64)
    pos16 = np.asarray(positive_y16_counts, dtype=np.float64)

    mean = image_counts.mean()
    var = image_counts.var(ddof=1)

    if var > mean:
        r_root = mean * mean / (var - mean)
    else:
        r_root = 1e6

    q50 = int(max(1, round(np.quantile(pos16, 0.50))))
    q85 = int(max(q50 + 1, round(np.quantile(pos16, 0.85))))

    return {
        "root_dispersion": float(r_root),
        "local_t1": q50,
        "local_t2": q85,
        "dense_threshold_16": q85,
    }
