"""Paired bootstrap over queries for AUPRC and recall at 99 % precision.

    python -m scripts.bootstrap [--boot 1000] [--scores cabid tmalign ...] [--raw DIR]

The same resampled query sets (default_rng(0)) are used for every ranking, so
differences between rankings can be read replicate by replicate. Resampling
queries only reweights the already sorted list, so each ranking is sorted once.

Writes cache/bootstrap/ci_b<B>.tsv and draws_b<B>.npz (row 0 = real data).
"""

import argparse
import time

import numpy as np
import pandas as pd

from cabid import pooled
from cabid.hits import load_hits, read_table, score
from cabid.paths import BOOTSTRAP, RAW
from cabid.scop import Scop
from cabid.style import label

LEVELS = ("family", "superfamily")
OTHER = {"tmalign": ("TMalign.txt", 2), "foldseek_tm": ("foldseekTM.txt", 10)}
DEFAULT = ["cabid", "cabid_sym", "bitdensity", "bitdensity_tgt", "bits", "tmalign",
           "foldseek_tm"]


def sorted_ranking(name, scop, raw):
    if name in OTHER:
        fn, col = OTHER[name]
        qi, ti, v = read_table(f"{raw}/{fn}", col, scop)
    else:
        h = load_hits()
        qi, ti, v = h["qi"], h["ti"], score(h, name)
    cat, w, _ = pooled.classify(scop, qi, ti)
    order = pooled.rank(v, False, ti)
    return qi[order], cat[order], w[order]


def replicates(q, c, w, ev, mults):
    """AUPRC and R@99%P for the real data (row 0) and each replicate."""
    n = w.size
    w_fp = np.where(c == pooled.FP, w, 0).astype(np.float32)
    w_tp = {lvl: np.where(c == pooled.CODE[lvl], w, 0).astype(np.float32) for lvl in LEVELS}
    m, fp, tp, prec = (np.empty(n) for _ in range(4))
    out = {(lvl, s): np.empty(len(mults) + 1) for lvl in LEVELS for s in ("auprc", "r99")}

    for b in range(len(mults) + 1):
        if b == 0:
            m[:] = 1.0
            nq = float(ev.sum())
        else:
            np.take(mults[b - 1], q, out=m)
            nq = float(mults[b - 1][ev].sum())
        np.multiply(w_fp, m, out=fp)
        np.cumsum(fp, out=fp)
        for lvl in LEVELS:
            np.multiply(w_tp[lvl], m, out=tp)
            np.cumsum(tp, out=tp)
            np.add(tp, fp, out=prec)
            np.maximum(prec, 1e-300, out=prec)
            np.divide(tp, prec, out=prec)
            tp /= nq
            out[(lvl, "auprc")][b] = float(np.dot(prec[1:], np.diff(tp)))
            cut = pooled.last_above(prec)
            out[(lvl, "r99")][b] = float(tp[cut]) if cut >= 0 else 0.0
        if b and b % 100 == 0:
            print(f"  replicate {b}/{len(mults)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--scores", nargs="*", default=DEFAULT)
    ap.add_argument("--raw", default=RAW)
    args = ap.parse_args()
    BOOTSTRAP.mkdir(parents=True, exist_ok=True)
    ci_path = BOOTSTRAP / f"ci_b{args.boot}.tsv"
    draw_path = BOOTSTRAP / f"draws_b{args.boot}.npz"

    scop = Scop()
    ev = pooled.evaluable(scop)
    rng = np.random.default_rng(0)
    mults = [np.bincount(rng.choice(scop.n, scop.n), minlength=scop.n).astype(float)
             for _ in range(args.boot)]

    draws = dict(np.load(draw_path)) if draw_path.exists() else {}
    rows = []
    for name in args.scores:
        t0 = time.time()
        d = replicates(*sorted_ranking(name, scop, args.raw), ev, mults)
        for (lvl, stat), v in d.items():
            draws[f"{name}|{lvl}|{stat}"] = v
            lo, hi = np.percentile(v[1:], [2.5, 97.5])
            rows.append({"key": name, "method": label(name), "level": lvl,
                         "statistic": stat, "point": v[0], "ci_lo": lo, "ci_hi": hi})
        print(f"{name}: {time.time() - t0:.0f} s", flush=True)

    table = pd.DataFrame(rows)
    if ci_path.exists():
        old = pd.read_csv(ci_path, sep="\t")
        table = pd.concat([old[~old.key.isin(args.scores)], table], ignore_index=True)
    table.to_csv(ci_path, sep="\t", index=False)
    np.savez_compressed(draw_path, **draws)


if __name__ == "__main__":
    main()
