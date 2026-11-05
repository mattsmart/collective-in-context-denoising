"""
Replot figures from the beta sweep npz produced by fig2_variance_decay_beta_run.py.

Figures produced (saved to ./fig2_output_variance_decay/):
    1. variance_decay_<mode>.png          — Var vs t, one color per beta,
                                            per-seed as thin transparent lines,
                                            mean as thick line, ±1σ shaded band
    2. variance_decay_<mode>_mean.png     — mean curves only (cleaner)
    3. mean_drift_<mode>.png              — cloud mean vs t (should stay near 0)

Edit the settings block at the top of main() to change paths or plot options.
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_OUT = os.path.join(SCRIPT_DIR, 'fig2_output_variance_decay')

# ---------------------------------------------------------------------------
# color palette flag
#   True  — qualitative palette (Tab10): distinct hues, good for small lists
#   False — Spectral: low beta → blue end, high beta → red end
# ---------------------------------------------------------------------------
BETA_CMAP_QUALITATIVE = False

def beta_colors(beta_list):
    n = len(beta_list)
    if BETA_CMAP_QUALITATIVE:
        cmap = plt.get_cmap('tab10')
        return [cmap(i % 10) for i in range(n)]
    else:
        cmap = plt.get_cmap('RdBu_r'); lo, hi = 0.0, 0.9
        return [cmap(lo + (hi - lo) * (i / max(n - 1, 1))) for i in range(n)]


def main():
    # =========================================================================
    # Settings — edit here
    # =========================================================================
    mode_label = 'mode1_rescale_eta'   # or 'mode2_fixed_eta' (mode 1 is theory/default)
    npz_path   = os.path.join(DIR_OUT, f'results_{mode_label}.npz')
    show_bands = True    # shade ±1 std around the mean
    show_seeds = True    # plot individual seed trajectories (thin, alpha)
    seed_alpha = 0.05    # transparency for per-seed lines
    beta_omit  = [5, 30, 40]   # beta values to skip in plots (too dense, but [] to include all)

    beta_zorder = {1: 5, 10: 20, 20: 25, 30: 24,
                   50: 4, 100: 8, 200:9}  # per-beta zorder override ({} for default)
    # =========================================================================

    print(f"Loading {npz_path}")
    data = np.load(npz_path)

    beta_list = [b for b in data['beta_list'].tolist() if b not in beta_omit]
    n_repeat  = int(data['n_repeat'])
    N         = int(data['N'])
    mu        = float(data['mu'])
    tau2      = float(data['tau2'])
    sigma2    = float(data['sigma2'])
    T_theory  = float(data['T_theory'])
    var_noisy = float(data['var_noisy'])
    var_clean = float(data['var_clean'])

    print(f"  beta_list={beta_list}  n_repeat={n_repeat}  N={N}")
    print(f"  tau2={tau2}  sigma2={sigma2}  var_noisy={var_noisy:.4f}")
    print(f"  T_theory={T_theory:.4f}")

    # unpack per-beta arrays
    results = {}
    for beta in beta_list:
        t_axis   = data[f'b{beta}_t']
        var_runs = data[f'b{beta}_var_runs']   # (n_repeat, L+1)
        std_runs = data[f'b{beta}_std_runs']
        mu_runs  = data[f'b{beta}_mu_runs']
        eta      = float(data[f'b{beta}_eta'])
        L        = int(data[f'b{beta}_L'])
        T_run    = float(data[f'b{beta}_T_run'])
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

    colors = beta_colors(beta_list)

    def zorder(beta, default=5):
        return beta_zorder.get(beta, default)

    # longest t axis for reference lines
    T_run_max = max(r['T_run'] for r in results.values())
    t_ref = np.linspace(0, T_run_max, 400)

    base_title = (f'β sweep  |  N={N}  τ²={tau2}  σ²={sigma2}  '
                  f'[{mode_label}]  n_repeat={n_repeat}')

    # -------------------------------------------------------------------------
    # figure 1: variance decay — per-seed + mean + band
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(4, 3.2))

    for color, beta in zip(colors, beta_list):
        r = results[beta]
        z = zorder(beta)
        if show_seeds:
            for ri in range(n_repeat):
                ax.plot(r['t_axis'], r['var_runs'][ri], '-', linewidth=0.6,
                        color=color, alpha=seed_alpha, zorder=z)
        if show_bands:
            ax.fill_between(r['t_axis'],
                            r['var_mean'] - r['var_std'],
                            r['var_mean'] + r['var_std'],
                            color=color, alpha=0.15, zorder=z)  # was 0.15 alpha
        ax.plot(r['t_axis'], r['var_mean'], '-', linewidth=2.0,
                color=color, label=f'β={beta}', zorder=z)

    ax.axhline(var_clean, color='deepskyblue', linestyle='--', linewidth=1.2,
               label=f'theory Var(x) = {var_clean:.3f}')
    ax.axhline(var_noisy, color='gray', linestyle='--', linewidth=1.2,
               label=f'theory Var(y⁰) = {var_noisy:.3f}')
    ax.axvline(T_theory, color='deepskyblue', linestyle=':', linewidth=1.5,
               label=f'T_theory = {T_theory:.3f}')
    ax.plot(t_ref, var_noisy - 2 * t_ref, 'k--', linewidth=1.5, zorder=100,
            label='linear ref:  Var(0) − 2t')

    ax.set_xlabel('Continuous time  t = l · η / β')
    ax.set_ylabel('Var(y⁽ˡ⁾)')
    ax.set_title(f'Variance decay vs β  (mean ± 1σ over {n_repeat} seeds)\n{base_title}')
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='-', linewidth=0.5, color='lightgray', alpha=0.5)
    ax.set_ylim(0.8 * var_clean, var_noisy * 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    for ext in ('png', 'svg'):
        p = os.path.join(DIR_OUT, f'variance_decay_{mode_label}.{ext}')
        fig.savefig(p, dpi=150)
        print(f"  Saved {p}")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # figure 2: mean curves only
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))

    for color, beta in zip(colors, beta_list):
        r = results[beta]
        ax.plot(r['t_axis'], r['var_mean'], '-', linewidth=1.8,
                color=color, label=f'β={beta}')

    ax.axhline(var_clean, color='deepskyblue', linestyle='--', linewidth=1.2,
               label=f'theory Var(x) = {var_clean:.3f}')
    ax.axhline(var_noisy, color='gray', linestyle='--', linewidth=1.2,
               label=f'theory Var(y⁰) = {var_noisy:.3f}')
    ax.axvline(T_theory, color='deepskyblue', linestyle=':', linewidth=1.5,
               label=f'T_theory = {T_theory:.3f}')
    ax.plot(t_ref, var_noisy - 2 * t_ref, 'k--', linewidth=1.0, zorder=100,
            label='linear ref:  Var(0) − 2t')

    ax.set_xlabel('Continuous time  t = l · η / β')
    ax.set_ylabel('Var(y⁽ˡ⁾)  [mean over seeds]')
    ax.set_title(f'Variance decay vs β  (mean over {n_repeat} seeds)\n{base_title}')
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
    ax.set_ylim(0.8 * var_clean, var_noisy * 1.05)
    plt.tight_layout()
    for ext in ('png', 'svg'):
        p = os.path.join(DIR_OUT, f'variance_decay_{mode_label}_mean.{ext}')
        fig.savefig(p, dpi=150)
        print(f"  Saved {p}")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # figure 3: mean drift (should stay near 0)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))

    for color, beta in zip(colors, beta_list):
        r = results[beta]
        ax.plot(r['t_axis'], r['mu_mean'], '-', linewidth=1.8,
                color=color, label=f'β={beta}', zorder=zorder(beta))

    ax.axhline(mu, color='green', linestyle='--', linewidth=1.2,
               label=f'prior mean μ={mu}')
    ax.axvline(T_theory, color='gray', linestyle=':', linewidth=1.2,
               label=f'T_theory = {T_theory:.3f}')

    ax.set_xlabel('Continuous time  t = l · η / β')
    ax.set_ylabel('Mean(y⁽ˡ⁾)  [mean over seeds]')
    ax.set_title(f'Cloud mean vs β  (mean over {n_repeat} seeds)\n{base_title}')
    ax.legend(fontsize=8)
    ax.grid(True)
    plt.tight_layout()
    for ext in ('png', 'svg'):
        p = os.path.join(DIR_OUT, f'mean_drift_{mode_label}.{ext}')
        fig.savefig(p, dpi=150)
        print(f"  Saved {p}")
    plt.close(fig)

    print(f"\nDone. Outputs in: {DIR_OUT}")


if __name__ == '__main__':
    main()