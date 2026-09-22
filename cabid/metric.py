import numpy as np

# a from the cross-validated sweep (scripts/cv_sweep.py), kappa from the
# logistic fit on superfamily labels (scripts/fit_kappa.py)
A_EXP = 0.65
KAPPA = 0.022599
KAPPA_SYM = 0.022815

MIN_COV = 70.0      # bitdensity coverage cut-off, in percent
PENALTY = -1e9      # score given to pairs below the cut-off (kept, ranked last)


def cabid(bits, qspan, tcov, a=A_EXP, k=KAPPA):
    """bits / qspan^a + k * tcov, with tcov in percent."""
    return np.asarray(bits, float) / np.asarray(qspan, float) ** a + k * np.asarray(tcov, float)


def cabid_sym(bits, qspan, tspan, qcov, tcov, a=A_EXP, k=KAPPA_SYM):
    """Smaller of the two orientations, so query and target are interchangeable."""
    bits = np.asarray(bits, float)
    q_side = bits / np.asarray(qspan, float) ** a + k * np.asarray(tcov, float)
    t_side = bits / np.asarray(tspan, float) ** a + k * np.asarray(qcov, float)
    return np.minimum(q_side, t_side)


def bitdensity(bits, qspan, qcov, tcov, min_cov=MIN_COV, rule="both"):
    """bits / qspan, with pairs under the coverage cut-off pushed to the bottom.

    rule="both" requires qcov and tcov >= min_cov, rule="target" only tcov.
    """
    v = np.asarray(bits, float) / np.asarray(qspan, float)
    low = np.asarray(tcov) < min_cov
    if rule == "both":
        low |= np.asarray(qcov) < min_cov
    elif rule != "target":
        raise ValueError("rule must be 'both' or 'target'")
    v[low] = PENALTY
    return v
