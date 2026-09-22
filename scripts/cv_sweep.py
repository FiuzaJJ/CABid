"""Grouped 5-fold cross-validation of the span exponent a (Figure S1).

    python -m scripts.cv_sweep [--grid 0.5 0.65 ...] [--out DIR]

SCOPe folds are dealt into five CV groups (largest first, into the smallest
group) and each hit follows its query. For every exponent and CV fold, kappa is
fitted on 3 M hits from the training folds, then six scores are evaluated on
the training folds (35 % of queries, used only to pick a) and on the held-out
fold. The held-out scores of all five folds are also pooled to count true
positives ahead of the n-th false positive.

About 20 minutes per exponent, so several hours for the full grid.
"""

import argparse
import time

import numpy as np
import pandas as pd

from cabid import logistic, pooled
from cabid.hits import load_hits
from cabid.paths import CV
from cabid.scop import Scop

K, SEED = 5, 11
FIT_ROWS = 3_000_000
TRAIN_QFRAC = 0.35
DEMOTED = np.float32(-1e18)
MODELS = ["Xa", "Xa_ktcov", "Xa_cov70", "sym_max", "sym_min", "sym_mean"]
GRID = np.round(np.concatenate([np.arange(0, 0.5, 0.1), np.arange(0.5, 1.0, 0.025),
                                np.arange(1.0, 1.55, 0.1)]), 6)


def cv_groups(scop, qi):
    hits_per_fold = np.bincount(scop.fold[qi], minlength=scop.fold.max() + 1)
    group = np.empty(hits_per_fold.size, np.int8)
    load = np.zeros(K)
    for f in np.argsort(-hits_per_fold):
        group[f] = np.argmin(load)
        load[group[f]] += hits_per_fold[f]
    return group[scop.fold]


def tp_before_nth_fp(score, is_tp, is_fp, ti, nths=range(1, 7)):
    """Same as pooled.tp_before_nth_fp on the ranked list, without sorting it."""
    s_fp, t_fp = score[is_fp], ti[is_fp]
    top = np.argpartition(-s_fp, max(nths) - 1)[:max(nths)]
    cand = s_fp >= s_fp[top].min()
    cs, ct = s_fp[cand], t_fp[cand]
    order = np.lexsort((ct, -cs))
    out = {}
    for n in nths:
        s_n, t_n = cs[order[n - 1]], ct[order[n - 1]]
        ahead = (score > s_n) | ((score == s_n) & (ti < t_n))
        out[n] = int(np.count_nonzero(is_tp & ahead))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=float, nargs="*", default=GRID)
    ap.add_argument("--out", default=CV)
    args = ap.parse_args()

    scop = Scop()
    h = load_hits()
    qi, ti = h["qi"], h["ti"]
    bits = h["bits"].astype(np.float32)
    qcov, tcov = h["qcov"], h["tcov"]
    logq = np.log(np.maximum(h["qspan"], 1)).astype(np.float32)
    logt = np.log(np.maximum(h["tspan"], 1)).astype(np.float32)
    covok = (qcov >= 70) & (tcov >= 70)
    y = (scop.sfam[qi] == scop.sfam[ti]).astype(float)
    cat, w, _ = pooled.classify(scop, qi, ti)
    is_fp = cat == pooled.FP
    is_tp = cat >= pooled.TP_FAM
    ev = pooled.evaluable(scop)
    del h

    group_of_dom = cv_groups(scop, qi)
    group = group_of_dom[qi]
    train_queries = np.random.default_rng(SEED).random(scop.n) < TRAIN_QFRAC
    in_train_sample = train_queries[qi]
    fit_rng = np.random.default_rng(SEED + 1)
    fit_idx = [np.sort(fit_rng.choice(np.flatnonzero(group != f), FIT_ROWS, replace=False))
               for f in range(K)]

    def kappas(f, a):
        i = fit_idx[f]
        xq = bits[i] * np.exp(-a * logq[i])
        xt = bits[i] * np.exp(-a * logt[i])
        b1 = logistic.fit(np.column_stack([xq, tcov[i]]), y[i])
        b2 = logistic.fit(np.column_stack([np.concatenate([xq, xt]),
                                           np.concatenate([tcov[i], qcov[i]])]),
                          np.concatenate([y[i], y[i]]))
        return b1[2] / b1[1], b2[2] / b2[1]

    def scores(i, a, k_lin, k_sym):
        xq = (bits[i] * np.exp(-a * logq[i])).astype(np.float32)
        xt = (bits[i] * np.exp(-a * logt[i])).astype(np.float32)
        q_side = xq + np.float32(k_sym) * tcov[i]
        t_side = xt + np.float32(k_sym) * qcov[i]
        return {"Xa": xq,
                "Xa_ktcov": (xq + np.float32(k_lin) * tcov[i]).astype(np.float32),
                "Xa_cov70": np.where(covok[i], xq, DEMOTED),
                "sym_max": np.maximum(q_side, t_side),
                "sym_min": np.minimum(q_side, t_side),
                "sym_mean": (0.5 * (q_side + t_side)).astype(np.float32)}

    def evaluate(i, s, nq):
        order = pooled.rank(np.asarray(s, float), False, ti[i])
        c, ww = cat[i][order], w[i][order]
        first = pooled.tp_before_nth_fp(c, (1,))[1]["tp_total"]
        return [{"level": lvl, **pooled.summary(*pooled.precision_recall(c, ww, nq, lvl)),
                 "tp_before_1st_global_fp": first} for lvl in pooled.LEVELS]

    train_rows, test_rows, fit_rows, nth_rows = [], [], [], []
    t0 = time.time()
    for a in args.grid:
        a = float(a)
        oof = {m: np.empty(qi.size, np.float32) for m in MODELS}
        for f in range(K):
            k_lin, k_sym = kappas(f, a)
            fit_rows.append({"cv_fold": f, "a": a, "k_lin": k_lin, "k_sym": k_sym})
            train = np.flatnonzero((group != f) & in_train_sample)
            test = np.flatnonzero(group == f)
            nq_train = int(((group_of_dom != f) & ev & train_queries).sum())
            nq_test = int(((group_of_dom == f) & ev).sum())
            for i, nq, sink in ((train, nq_train, train_rows), (test, nq_test, test_rows)):
                for m, s in scores(i, a, k_lin, k_sym).items():
                    sink += [{"cv_fold": f, "a": a, "model": m, **r} for r in evaluate(i, s, nq)]
                    if sink is test_rows:
                        oof[m][i] = s
        for m, s in oof.items():
            nth_rows += [{"a": a, "model": m, "nth": n, "tp_pooled": v}
                         for n, v in tp_before_nth_fp(s, is_tp, is_fp, ti).items()]
        print(f"a = {a:<6g} {(time.time() - t0) / 60:.0f} min", flush=True)

    for name, rows in (("cv_train_curves", train_rows), ("cv_test_curves", test_rows),
                       ("cv_fits", fit_rows), ("cv_pooled_tp_nth", nth_rows)):
        pd.DataFrame(rows).to_csv(f"{args.out}/{name}.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
