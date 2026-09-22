"""Coverage coefficient of CABid from a logistic fit on all Foldseek hits.

    python -m scripts.fit_kappa [--a 0.65]

    logit P(same superfamily) = c + b1 * bits/span^a + b2 * cov,   kappa = b2 / b1

The symmetric fit stacks both orientations of every hit with the same label.
"""

import argparse

import numpy as np

from cabid import logistic
from cabid.hits import load_hits
from cabid.scop import Scop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=float, default=0.65)
    a = ap.parse_args().a

    scop = Scop()
    h = load_hits()
    y = (scop.sfam[h["qi"]] == scop.sfam[h["ti"]]).astype(float)
    xq = h["bits"] / h["qspan"] ** a
    xt = h["bits"] / h["tspan"] ** a

    b = logistic.fit(np.column_stack([xq, h["tcov"]]), y)
    print(f"kappa     = {b[2] / b[1]:.6f}   (bits/qspan^a + kappa*tcov)")

    X = np.column_stack([np.concatenate([xq, xt]), np.concatenate([h["tcov"], h["qcov"]])])
    b = logistic.fit(X, np.concatenate([y, y]))
    print(f"kappa_sym = {b[2] / b[1]:.6f}   (min over both orientations)")


if __name__ == "__main__":
    main()
