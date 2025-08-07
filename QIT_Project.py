import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigvalsh
from scipy.stats import unitary_group


def trace_distance(rho, sigma):
    # 0.5 * Tr|ρ-σ|  (for Hermitian matrices the 1-norm of eigenvalues = sum |λ_i|)
    evals = eigvalsh(rho - sigma)
    return 0.5 * np.sum(np.abs(evals))


def main():
    # --- fixed spectra ---
    p = np.array([0.55, 0.25, 0.15, 0.05])
    q = np.array([0.40, 0.30, 0.25, 0.05])
    N = len(p)
    Dmin = trace_distance(np.diag(np.sort(p)), np.diag(np.sort(q)))        # p↑ , q↑
    Dmax = trace_distance(np.diag(np.sort(p)), np.diag(np.sort(q)[::-1]))  # p↑ , q↓

    # Monte-Carlo distances
    n_samp = 20000
    dvals = []
    for _ in range(n_samp):
        U = unitary_group.rvs(N)
        V = unitary_group.rvs(N)
        rho = U @ np.diag(p) @ U.conj().T
        sig = V @ np.diag(q) @ V.conj().T
        dvals.append(trace_distance(rho, sig))

    # optional: include the two diagonal cases themselves
    dvals += [Dmin, Dmax]

    # --- plot ---
    plt.hist(dvals, bins=70, alpha=0.7, color='steelblue')
    plt.axvline(Dmin, color='red',   lw=2, label=r"$D_{\min}$")
    plt.axvline(Dmax, color='green', lw=2, label=r"$D_{\max}$")
    plt.xlabel("Trace Distance")
    plt.ylabel("Frequency")
    title_str = (
        "Empirical Trace‐Distance Distribution for Unitary Orbits (N=4)\n"
        f"p = {p},   q = {q}"
    )
    plt.title(title_str, fontsize=10)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
