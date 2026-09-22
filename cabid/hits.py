import json
from array import array

import numpy as np

from .metric import bitdensity, cabid, cabid_sym
from .paths import DOMAIN_LENGTHS, HIT_COLUMNS
from .scop import Scop


def domain_lengths():
    """Residues per domain (from Foldseek self-hits). Missing domains get 1."""
    L = np.load(DOMAIN_LENGTHS).astype(float)
    L[L == 0] = 1.0
    return L


def load_hits():
    """All non-self Foldseek hits between SCOPe40 domains, from the column cache."""
    meta_path = HIT_COLUMNS / "meta.json"
    if not meta_path.exists():
        raise SystemExit("no hit columns found, run scripts/extract_hits.py first")
    meta = json.loads(meta_path.read_text())

    def col(name):
        return np.fromfile(HIT_COLUMNS / f"{name}.bin", dtype=meta["columns"][name])

    qi, ti = col("qi").astype(np.int32), col("ti").astype(np.int32)
    qspan, tspan = col("qspan").astype(float), col("tspan").astype(float)
    L = domain_lengths()
    return {"n": meta["n_hits"], "qi": qi, "ti": ti, "bits": col("bits").astype(float),
            "qspan": qspan, "tspan": tspan,
            "qcov": 100 * qspan / L[qi], "tcov": 100 * tspan / L[ti]}


def score(h, name):
    """Foldseek-derived rankings over the hit table. Higher is better."""
    if name == "bits":
        return h["bits"].copy()
    if name == "cabid":
        return cabid(h["bits"], h["qspan"], h["tcov"])
    if name == "cabid_sym":
        return cabid_sym(h["bits"], h["qspan"], h["tspan"], h["qcov"], h["tcov"])
    if name in ("bitdensity", "bitdensity_tgt"):
        rule = "both" if name == "bitdensity" else "target"
        return bitdensity(h["bits"], h["qspan"], h["qcov"], h["tcov"], rule=rule)
    raise KeyError(name)


def read_table(path, col, scop=None, dtype="f"):
    """Stream a tab-separated result table into (qi, ti, value).

    Self-hits and ids that are not in the SCOPe lookup are skipped.
    """
    k2i = (scop or Scop()).key2idx
    qs, ts, vs = array("i"), array("i"), array(dtype)
    with open(path, buffering=1 << 22) as fh:
        for line in fh:
            f = line.split("\t")
            q, t = k2i.get(f[0], -1), k2i.get(f[1], -1)
            if q < 0 or t < 0 or q == t:
                continue
            qs.append(q)
            ts.append(t)
            vs.append(float(f[col]))
    return (np.frombuffer(qs, np.int32).copy(), np.frombuffer(ts, np.int32).copy(),
            np.frombuffer(vs, np.float32 if dtype == "f" else np.float64).copy())
