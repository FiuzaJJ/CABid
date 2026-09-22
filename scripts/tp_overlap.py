"""Superfamily true positives each ranking retrieves at its own 99 % precision cut.

    python -m scripts.tp_overlap [--raw DIR]

CABid symmetric, bits and bitdensity rank the Foldseek hits. TMalign ranks its
own all-against-all list; its true positives are then looked up in the Foldseek
hit table (every one of them is a Foldseek hit), so all four sets are row
indices of the same table.

Writes cache/overlap/tp_overlap_superfamily.npz and tp_points_superfamily.npz.
"""

import argparse

import numpy as np

from cabid import pooled
from cabid.hits import load_hits, read_table, score
from cabid.paths import OVERLAP, RAW
from cabid.scop import Scop

LEVEL = "superfamily"
FOLDSEEK = {"CABid symmetric": "cabid_sym", "bits": "bits", "bitdensity": "bitdensity"}


def tps_at_p99(v, qi, ti, scop):
    """Rows of the level's TPs above the 99 % precision cut, the cut and the recall."""
    cat, w, nq = pooled.classify(scop, qi, ti)
    order = pooled.rank(v, False, ti)
    c = cat[order]
    prec, rec = pooled.precision_recall(c, w[order], nq, LEVEL)
    cut = pooled.last_above(prec)
    rows = order[:cut + 1][c[:cut + 1] == pooled.CODE[LEVEL]]
    return np.sort(rows), cut, float(rec[cut]) if cut >= 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=RAW)
    args = ap.parse_args()
    OVERLAP.mkdir(parents=True, exist_ok=True)

    scop = Scop()
    h = load_hits()
    out = {"n_hits": h["n"]}
    for name, key in FOLDSEEK.items():
        rows, cut, rec = tps_at_p99(score(h, key), h["qi"], h["ti"], scop)
        out[f"rows|{name}"], out[f"cut|{name}"], out[f"recall|{name}"] = rows, cut, rec
        print(f"{name:<16} {rows.size:>6,} TPs  recall@99%P {rec:.4f}", flush=True)

    qi, ti, v = read_table(f"{args.raw}/TMalign.txt", 2, scop)
    rows, cut, rec = tps_at_p99(v, qi, ti, scop)
    print(f"{'TMalign':<16} {rows.size:>6,} TPs  recall@99%P {rec:.4f}", flush=True)
    tm_keys = np.sort(qi[rows].astype(np.int64) * scop.n + ti[rows])
    out["tmalign_n_pairs"] = qi.size
    del qi, ti, v

    fs_keys = h["qi"].astype(np.int64) * scop.n + h["ti"]
    by_key = np.argsort(fs_keys)
    pos = np.searchsorted(fs_keys[by_key], tm_keys)
    assert np.array_equal(fs_keys[by_key][pos], tm_keys), "TMalign TP missing from Foldseek hits"
    out["rows|TMalign"], out["cut|TMalign"], out["recall|TMalign"] = np.sort(by_key[pos]), cut, rec
    np.savez_compressed(OVERLAP / f"tp_overlap_{LEVEL}.npz", **out)

    # span and coverage of every retrieved TP, for the scatter panels
    u = np.unique(np.concatenate([out[f"rows|{n}"] for n in [*FOLDSEEK, "TMalign"]]))
    np.savez_compressed(OVERLAP / f"tp_points_{LEVEL}.npz", rows=u,
                        qspan=h["qspan"][u].astype(np.float32),
                        tcov=h["tcov"][u].astype(np.float32))


if __name__ == "__main__":
    main()
