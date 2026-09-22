"""Read foldseek.txt once and cache what the other scripts need.

    python -m scripts.extract_hits [--raw DIR]

Writes
    cache/hit_columns/    qi, ti, qspan, tspan, bits, evalue for every non-self hit
    cache/self/scope40_self_alignments.tsv
    data/domain_lengths.npy   qend of the domain's self-hit, or the furthest
                              tend it reaches as a target if it has none

qspan and tspan are qend-qstart+1 and tend-tstart+1 (gaps included).
"""

import argparse
import json
from array import array

import numpy as np

from cabid.paths import DOMAIN_LENGTHS, HIT_COLUMNS, RAW, SELF
from cabid.scop import Scop

COLS = {"qi": "h", "ti": "h", "qspan": "H", "tspan": "H", "bits": "f", "evalue": "d"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=RAW)
    args = ap.parse_args()

    scop = Scop()
    k2i = scop.key2idx
    HIT_COLUMNS.mkdir(parents=True, exist_ok=True)
    SELF.mkdir(parents=True, exist_ok=True)

    bufs = {c: array(t) for c, t in COLS.items()}
    files = {c: open(HIT_COLUMNS / f"{c}.bin", "wb") for c in COLS}
    lengths = np.zeros(scop.n, np.int32)
    max_tend = np.zeros(scop.n, np.int32)
    n_rows = n_self = n_hits = 0

    with open(f"{args.raw}/foldseek.txt", buffering=1 << 22) as fh, \
            open(SELF / "scope40_self_alignments.tsv", "w") as self_out:
        self_out.write("domain\talnlen\tevalue\tbits\n")
        for line in fh:
            f = line.split("\t")
            n_rows += 1
            q, t = k2i.get(f[0], -1), k2i.get(f[1], -1)
            if q < 0 or t < 0:
                continue
            max_tend[t] = max(max_tend[t], int(f[9]))
            if q == t:
                n_self += 1
                lengths[q] = int(f[7])
                self_out.write(f"{f[0]}\t{f[3]}\t{float(f[10])}\t{float(f[11])}\n")
                continue
            bufs["qi"].append(q)
            bufs["ti"].append(t)
            bufs["qspan"].append(int(f[7]) - int(f[6]) + 1)
            bufs["tspan"].append(int(f[9]) - int(f[8]) + 1)
            bufs["bits"].append(float(f[11]))
            bufs["evalue"].append(float(f[10]))
            n_hits += 1
            if len(bufs["qi"]) >= 4_000_000:
                for c, b in bufs.items():
                    b.tofile(files[c])
                    del b[:]
    for c, b in bufs.items():
        b.tofile(files[c])
        files[c].close()

    np.save(DOMAIN_LENGTHS, np.where(lengths > 0, lengths, max_tend))
    meta = {"n_rows": n_rows, "n_self_hits": n_self, "n_hits": n_hits, "columns": COLS}
    (HIT_COLUMNS / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
