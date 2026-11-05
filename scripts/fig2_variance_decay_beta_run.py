"""
Variance decay sweep over beta — tests whether cloud variance decreases
linearly with the effective denoising time t = l * stepsize = l * eta / beta.

Theory prediction: in the large-N, small-h limit the leading-order variance dynamics satisfy

    Var(y(t)) ≈ tau2 + sigma2 - 2 * t   for t in [0, sigma2/2], with Var(y(T)) ≈ tau2
    d/dt Var(y(t)) ≈ - 2                for t in [0, sigma2/2]

so plotting Var(y^(l)) vs t = l * eta / beta should collapse curves
from different beta values onto a single line for small t.

Two modes (set in the settings block):
    mode_1_rescale_eta = True
        For each beta, set eta = beta * sigma2 / (2 * L_theory) so that
        the predicted optimal stopping time is always T = sigma2/2.
        Run for L = L_factor * L_theory layers (t runs to L_factor * sigma2/2).
        Curves are comparable on the shared continuous-time axis t = l * eta / beta.

    mode_1_rescale_eta = False
        Fix eta and L regardless of beta.  Different beta values get
        different effective integration times and kernel widths.

Each beta is run for n_repeat independent seeds and results are averaged.
All raw trajectories are saved to an npz; run fig2_variance_decay_beta_replot.py to plot.
"""

import os
import time

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_OUT = os.path.join(SCRIPT_DIR, 'fig2_output_variance_decay')
os.makedirs(DIR_OUT, exist_ok=True)


# ---------------------------------------------------------------------------
# lightweight trajectory runner (numpy only, no JAX) for quick testing and data collection
# ---------------------------------------------------------------------------

def gauss_attn_1d(y, beta):
    """
    Gaussian RBF self-attention for 1D particle cloud.
    y: (N,) float
    Returns attn_out: (N,)  where attn_out[i] = sum_j a_ij * y[j]
    a_ij = softmax_j(-beta/2 * (y_i - y_j)^2)
    """
    diff_sq = (y[:, None] - y[None, :]) ** 2
    logits = -0.5 * beta * diff_sq
    logits -= logits.max(axis=1, keepdims=True)
    w = np.exp(logits)
    w /= w.sum(axis=1, keepdims=True)
    return w @ y


def run_moments(z0, beta, eta, L):
    """
    Run L layers and record empirical mean/std/var at every step.

    Returns:
        t_axis : (L+1,)  continuous time  t[l] = l * eta / beta
        mu_t   : (L+1,)
        std_t  : (L+1,)
        var_t  : (L+1,)
    """
    h = eta / beta
    z = z0.astype(np.float64)

    t_axis = np.empty(L + 1)
    mu_t   = np.empty(L + 1)
    std_t  = np.empty(L + 1)
    var_t  = np.empty(L + 1)

    t_axis[0] = 0.0
    mu_t[0]   = float(np.mean(z))
    std_t[0]  = float(np.std(z))
    var_t[0]  = float(np.var(z))

    for l in range(1, L + 1):
        attn_out = gauss_attn_1d(z, beta)
        z = (1.0 - eta) * z + eta * attn_out
        t_axis[l] = l * h
        mu_t[l]   = float(np.mean(z))
        std_t[l]  = float(np.std(z))
        var_t[l]  = float(np.var(z))

    return t_axis, mu_t, std_t, var_t


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    # =========================================================================
    # Settings — edit here
    # =========================================================================
    N      = 5000     # figure 2 uses N=5000
    mu     = 0.0
    tau2   = 1.0      # prior variance  (defines 'clean target')
    sigma2 = 0.25     # noise variance  (sigma^2)

    # Beta sweep
    beta_list = [1, 5, 10, 20, 30, 40, 50, 100, 200]

    # Repeat settings
    n_repeat  = 20     # independent seeds per beta (runtime-- approx. 1 minute per seed)
    seed_base = 0      # seed for repeat r is seed_base + r * 1000

    # Mode flag
    mode_1_rescale_eta = True   # True = rescale eta per beta; False = fixed eta/L (theory --> True)

    # Mode 1 settings
    L_theory_base = 200   # reference depth; eta = beta*sigma2 / (2*L_theory_base) — from theory
    L_factor      = 2     # run to this multiple of L_theory  (T_run = L_factor * sigma2/2)

    # Mode 2 settings (used only when mode_1_rescale_eta=False)
    L_fixed   = 400
    eta_fixed = 0.25
    # =========================================================================

    var_noisy = tau2 + sigma2
    var_clean = tau2
    T_theory  = sigma2 / 2.0  # optimal stopping time according to theory (drop higher-order corrections)

    mode_label = 'mode1_rescale_eta' if mode_1_rescale_eta else 'mode2_fixed_eta'
    print(f"\n=== Variance decay sweep over beta  ({mode_label}) ===")
    print(f"N={N}  tau2={tau2}  sigma2={sigma2}")
    print(f"T_theory = sigma2/2 = {T_theory:.4f}")
    print(f"beta_list = {beta_list}   n_repeat = {n_repeat}\n")

    results = {}   # beta -> dict

    for beta in beta_list:
        if mode_1_rescale_eta:
            L_theory = L_theory_base
            eta = beta * sigma2 / (2 * L_theory)
            if eta >= 1.0:
                L_theory = int(np.ceil(beta * sigma2 / (2 * 0.9)))
                eta = beta * sigma2 / (2 * L_theory)
                print(f"  beta={beta}: clamped L_theory={L_theory} to keep eta<1")
            L    = L_factor * L_theory
            T_run = L * eta / beta
        else:
            eta  = eta_fixed
            L    = L_fixed
            T_run = L * eta / beta

        h = eta / beta
        print(f"  beta={beta:4d}  eta={eta:.5f}  h={h:.6f}  L={L}  T_run={T_run:.4f}")

        var_runs = np.empty((n_repeat, L + 1))
        std_runs = np.empty((n_repeat, L + 1))
        mu_runs  = np.empty((n_repeat, L + 1))
        t_axis   = None
        t_beta_start = time.time()

        for r in range(n_repeat):
            seed = seed_base + r * 1000
            rng  = np.random.default_rng(seed)
            x_clean = rng.normal(mu, np.sqrt(tau2), size=N)
            noise   = rng.normal(0.0, np.sqrt(sigma2), size=N)
            z0      = x_clean + noise

            t_seed_start = time.time()
            t_ax, mu_t, std_t, var_t = run_moments(z0, beta=beta, eta=eta, L=L)
            print(f"    seed {r+1:2d}/{n_repeat}  {time.time() - t_seed_start:.1f}s")
            var_runs[r] = var_t
            std_runs[r] = std_t
            mu_runs[r]  = mu_t
            if t_axis is None:
                t_axis = t_ax   # same for all repeats at this beta

        print(f"  beta={beta:4d} done  total={time.time() - t_beta_start:.1f}s")

        results[beta] = dict(
            t_axis=t_axis,
            var_runs=var_runs,
            std_runs=std_runs,
            mu_runs=mu_runs,
            var_mean=var_runs.mean(axis=0),
            var_std =var_runs.std(axis=0),
            std_mean=std_runs.mean(axis=0),
            mu_mean =mu_runs.mean(axis=0),
            eta=eta, L=L, T_run=T_run,
        )

    # -------------------------------------------------------------------------
    # save all raw data to npz
    # -------------------------------------------------------------------------
    save_dict = dict(
        beta_list=np.array(beta_list),
        n_repeat=n_repeat,
        N=N, mu=mu, tau2=tau2, sigma2=sigma2,
        T_theory=T_theory, var_noisy=var_noisy, var_clean=var_clean,
        mode_1_rescale_eta=mode_1_rescale_eta,
        L_theory_base=L_theory_base, L_factor=L_factor,
        seed_base=seed_base,
    )
    for beta in beta_list:
        r = results[beta]
        save_dict[f'b{beta}_t']        = r['t_axis']
        save_dict[f'b{beta}_var_runs'] = r['var_runs']   # (n_repeat, L+1)
        save_dict[f'b{beta}_std_runs'] = r['std_runs']
        save_dict[f'b{beta}_mu_runs']  = r['mu_runs']
        save_dict[f'b{beta}_eta']      = r['eta']
        save_dict[f'b{beta}_L']        = r['L']
        save_dict[f'b{beta}_T_run']    = r['T_run']

    npz_path = os.path.join(DIR_OUT, f'results_{mode_label}.npz')
    np.savez(npz_path, **save_dict)
    print(f"\n  Saved raw data to {npz_path}")
    print(f"  Run fig2_variance_decay_beta_replot.py to produce figures.")
    print(f"\nDone.")


if __name__ == '__main__':
    main()