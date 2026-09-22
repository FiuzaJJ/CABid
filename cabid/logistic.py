import numpy as np
from scipy.special import expit


def fit(X, y, tol=1e-10, max_iter=100, chunk=4_000_000):
    """Logistic regression by Newton-Raphson. Returns [intercept, *coefficients].

    Predictors are standardised while fitting and mapped back at the end. The
    gradient and Hessian are summed in chunks so large tables fit in memory.
    """
    X = np.asarray(X, float)
    n, k = X.shape
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    b = np.zeros(k + 1)
    for _ in range(max_iter):
        g, H = np.zeros(k + 1), np.zeros((k + 1, k + 1))
        for lo in range(0, n, chunk):
            Z = np.column_stack([np.ones(min(chunk, n - lo)), (X[lo:lo + chunk] - mu) / sd])
            p = expit(Z @ b)
            g += Z.T @ (y[lo:lo + chunk] - p)
            H += (Z * (p * (1 - p))[:, None]).T @ Z
        step = np.linalg.solve(H, g)
        b += step
        if np.abs(step).max() < tol:
            break
    return np.concatenate([[b[0] - np.sum(b[1:] * mu / sd)], b[1:] / sd])
