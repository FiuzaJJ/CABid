"""Pooled SCOPe benchmark: every alignment of every query in one ranked list.

Same labels and weights as bench.fdr.noselfhit.awk from foldseek-analysis:
a hit to another fold is a false positive, and each hit counts 1/norm, where
norm is the number of relatives the query has at that level.

    precision = tp / (tp + fp)
    recall    = tp / number of evaluable queries
"""

import numpy as np

FP, SKIP, TP_FAM, TP_SFAM, TP_FOLD = 0, 1, 2, 3, 4
LEVELS = ("family", "superfamily", "fold")
CODE = {"family": TP_FAM, "superfamily": TP_SFAM, "fold": TP_FOLD}


def evaluable(scop):
    """Queries with at least one relative at family, superfamily and fold level."""
    fam1, sfam1, fold1 = scop.n_fam - 1, scop.n_sfam - 1, scop.n_fold - 1
    return (fam1 != 0) & (sfam1 != fam1) & (fold1 != sfam1)


def classify(scop, qi, ti):
    """Category and 1/norm weight of every pair (qi, ti)."""
    fam1, sfam1, fold1 = scop.n_fam - 1, scop.n_sfam - 1, scop.n_fold - 1
    ev = evaluable(scop)

    same_fold = scop.fold[qi] == scop.fold[ti]
    same_sfam = scop.sfam[qi] == scop.sfam[ti]
    same_fam = scop.fam[qi] == scop.fam[ti]

    cat = np.full(qi.size, SKIP, np.int8)
    cat[~same_fold] = FP
    ok = same_fold & ev[qi]
    cat[ok & same_fam] = TP_FAM
    cat[ok & ~same_fam & same_sfam] = TP_SFAM
    cat[ok & ~same_fam & ~same_sfam] = TP_FOLD

    fp_norm = np.maximum(fold1 - sfam1, fam1)
    fp_norm[fp_norm == 0] = 1
    norm = {FP: fp_norm, TP_FAM: fam1, TP_SFAM: sfam1 - fam1, TP_FOLD: fold1 - sfam1}
    w = np.zeros(qi.size)
    for code, n in norm.items():
        m = cat == code
        w[m] = 1.0 / n[qi[m]]
    return cat, w, int(ev.sum())


def rank(score, ascending, ti):
    """Order of the global list. Ties go to the smaller target index."""
    return np.lexsort((ti, score if ascending else -score))


def precision_recall(c, w, n_queries, level, fp=None):
    """Precision and recall down a ranked list (c, w already in rank order).

    `fp` is the cumulative false-positive weight; pass it in when calling this
    for several levels on the same list.
    """
    if fp is None:
        fp = np.cumsum(np.where(c == FP, w, 0), dtype=np.float64)
    tp = np.cumsum(np.where(c == CODE[level], w, 0), dtype=np.float64)
    prec = tp + fp
    lead = int(np.searchsorted(prec, 0.0, side="right"))  # nothing retrieved yet
    np.maximum(prec, 1e-300, out=prec)
    np.divide(tp, prec, out=prec)
    prec[:lead] = 1.0
    tp /= n_queries
    return prec, tp


def summary(prec, rec):
    out = {"auprc": float(np.trapezoid(prec, rec)), "max_recall": float(rec[-1])}
    for p in (0.99, 0.95, 0.90):
        ok = prec >= p
        out[f"recall@P{int(p * 100)}"] = float(rec[ok].max()) if ok.any() else 0.0
    return out


def last_above(prec, p=0.99):
    """Deepest rank still at precision >= p, or -1."""
    ok = np.flatnonzero(prec >= p)
    return int(ok[-1]) if ok.size else -1


def tp_before_nth_fp(c, nths=(1, 2, 3, 4, 5, 6)):
    """Raw counts of same-fold hits ranked above the n-th false positive."""
    k, fps, block = max(nths), [], 1 << 22
    for start in range(0, c.size, block):  # the first FPs come early; avoid a full scan
        fps.extend(np.flatnonzero(c[start:start + block] == FP)[:k - len(fps)] + start)
        if len(fps) == k:
            break
    fps = np.array(fps, np.int64)
    out = {}
    for n in nths:
        stop = fps[n - 1] if n <= fps.size else c.size
        head = c[:stop]
        out[n] = {"tp_total": int(np.count_nonzero(head >= TP_FAM)),
                  "tp_family": int(np.count_nonzero(head == TP_FAM)),
                  "rank": int(stop)}
    return out
