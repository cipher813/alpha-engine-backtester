"""
stats_utils.py — shared statistical helpers for backtester analysis modules.

Benjamini-Hochberg FDR correction for multiple comparison correction across
correlated financial time-series tests.
"""


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """
    Benjamini-Hochberg procedure for controlling the false discovery rate.

    Args:
        p_values: List of raw p-values from multiple hypothesis tests.
        alpha: Target FDR level (default 0.05).

    Returns:
        List of booleans — True if the corresponding test is significant
        after FDR correction, False otherwise.

    BH is preferred over Bonferroni for correlated financial tests because
    Bonferroni controls family-wise error rate (too conservative with
    correlated tests) while BH controls false discovery rate.
    """
    n = len(p_values)
    if n == 0:
        return []

    # Pair each p-value with its original index, sort ascending
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    # Find the largest k where p_(k) <= (k/n) * alpha
    significant = [False] * n
    max_k = -1
    for rank_minus_one, (orig_idx, p) in enumerate(indexed):
        k = rank_minus_one + 1  # 1-based rank
        threshold = (k / n) * alpha
        if p <= threshold:
            max_k = rank_minus_one

    # All tests with rank <= max_k are significant
    if max_k >= 0:
        for i in range(max_k + 1):
            orig_idx = indexed[i][0]
            significant[orig_idx] = True

    return significant
