"""Pooled precision-recall, TPs before the n-th FP and per-class statistics.

    python -m scripts.pooled_curves [--raw DIR] [--only foldseek tmalign ...]

Each tool is scored on all the alignments it reports; nothing is filtered.
Writes cache/pooled/<tool>.npz.
"""

import argparse

import numpy as np

from cabid import pooled
from cabid.hits import load_hits, read_table, score
from cabid.paths import HIT_COLUMNS, POOLED, RAW
from cabid.scop import Scop

FOLDSEEK_SCORES = ["bits", "bitdensity", "bitdensity_tgt", "cabid", "cabid_sym"]
# tool -> (file, score column)
OTHER = {"foldseek_tm": ("foldseekTM.txt", 10), "tmalign": ("TMalign.txt", 2),
         "dali": ("dali.txt", 2)}


def thin(n, k=6000):
    """Points kept for plotting: every rank at the head, log-spaced after."""
    if n <= k:
        return np.arange(n)
    head = np.arange(2000)
    tail = np.geomspace(2000, n, k - 2000).astype(np.int64) - 1
    return np.unique(np.concatenate([head, tail]))


def analyse(order, cat, w, qi, n_queries, cls, ev):
    c, ww, q = cat[order], w[order], qi[order]
    curves, stats, cls_stats = {}, {}, {}

    fp = np.cumsum(np.where(c == pooled.FP, ww, 0), dtype=np.float64)
    keep = thin(c.size)
    for lvl in pooled.LEVELS:
        prec, rec = pooled.precision_recall(c, ww, n_queries, lvl, fp)
        stats[lvl] = pooled.summary(prec, rec)
        curves[lvl] = (prec[keep].astype(np.float32), rec[keep].astype(np.float32))
        del prec, rec
    del fp
    nth = pooled.tp_before_nth_fp(c)

    # restricting the global list to one class keeps that class's own order
    for cl in np.unique(cls):
        nq = int(((cls == cl) & ev).sum())
        sel = (cls == cl)[q]
        if nq == 0 or not sel.any():
            continue
        cs, cw = c[sel], ww[sel]
        cls_stats[(cl, "-", "n_queries")] = nq
        cls_stats[(cl, "-", "n_pairs")] = cs.size
        cfp = np.cumsum(np.where(cs == pooled.FP, cw, 0), dtype=np.float64)
        for lvl in pooled.LEVELS:
            for k, v in pooled.summary(*pooled.precision_recall(cs, cw, nq, lvl, cfp)).items():
                cls_stats[(cl, lvl, k)] = v
        cls_stats[(cl, "nth1", "tp_total")] = pooled.tp_before_nth_fp(cs, (1,))[1]["tp_total"]
    return curves, stats, nth, cls_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=RAW)
    ap.add_argument("--only", nargs="*", default=["foldseek", *OTHER])
    args = ap.parse_args()
    POOLED.mkdir(parents=True, exist_ok=True)

    scop = Scop()
    ev = pooled.evaluable(scop)

    for tool in args.only:
        if tool == "foldseek":
            h = load_hits()
            qi, ti = h["qi"], h["ti"]
            builders = {s: (lambda s=s: score(h, s), False) for s in FOLDSEEK_SCORES}
            ev_col = np.fromfile(HIT_COLUMNS / "evalue.bin", np.float64)
            builders["evalue"] = (lambda: ev_col, True)
        else:
            fn, col = OTHER[tool]
            qi, ti, v = read_table(f"{args.raw}/{fn}", col, scop)
            builders = {tool: (lambda v=v: v.astype(float), False)}

        cat, w, nq = pooled.classify(scop, qi, ti)
        w = w.astype(np.float32)
        payload, stats, nth, cls_stats = {}, {}, {}, {}
        for name, (build, ascending) in builders.items():
            order = pooled.rank(build(), ascending, ti)
            cur, st, nt, cs = analyse(order, cat, w, qi, nq, scop.cls_name, ev)
            del order
            for lvl in pooled.LEVELS:
                payload[f"{name}|{lvl}|prec"], payload[f"{name}|{lvl}|rec"] = cur[lvl]
                stats.update({f"{name}|{lvl}|{k}": v for k, v in st[lvl].items()})
            for n, d in nt.items():
                nth.update({f"{name}|{n}|{k}": v for k, v in d.items()})
            cls_stats.update({f"{name}|{cl}|{lvl}|{k}": v for (cl, lvl, k), v in cs.items()})
            print(f"{name:<15} family AUPRC {st['family']['auprc']:.4f}  "
                  f"R@99%P {st['family']['recall@P99']:.4f}", flush=True)

        np.savez_compressed(
            POOLED / f"{tool}.npz", n_pairs=qi.size, n_queries=nq,
            stat_keys=list(stats), stat_vals=list(stats.values()),
            nth_keys=list(nth), nth_vals=np.array(list(nth.values()), float),
            cls_keys=list(cls_stats), cls_vals=np.array(list(cls_stats.values()), float),
            **payload)


if __name__ == "__main__":
    main()
