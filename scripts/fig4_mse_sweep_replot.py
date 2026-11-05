"""
Replot figures from the parameter sweep npz produced by fig4_mse_sweep_run.py.

Wasserstein note:
    'w1' stored in the npz is SLICED W1 (fast proxy, 50 random projections).
    'w2' is exact W2 via POT EMD (subsampled to 300 points).
    If run_full_wasserstein=True was set in the sweep run, 'w1_full' / 'w2_full'
    (exact W1 via LP, subsampled) are also available.

MSE baselines:
    mse_init          — noisy input (l=0), finite-N empirical
    mse_s1_opt        — best Stage 1 cloud (min over depth), finite-N
    mse_s2_opt        — best Stage 2 MSE (min of early_s2_cross_beta over depth), finite-N
    mse_s2            — Stage 2 final cross-attention at l=L, finite-N
    mse_tweedie       — Stage 2 only (no prior refinement), using the legacy data key:
                        gauss_cross_attn(z0, z0, beta_final); finite-N
    bayes_mse         — finite-N empirical Bayes MMSE (seed-specific)
    analytic_bayes_mse— near-analytic Bayes MMSE (20k-sample draw, ~deterministic)

Figures produced (saved to ./fig4_sweep_output/):
    1. mse_vs_<axis>.png/svg          — all MSE curves vs sweep axis
    2. w1w2_vs_<axis>.png/svg         — W1 (sliced) / W2 at Stage 2 vs sweep axis
    3. mse_traj_panels_<axis>.png     — per-config Stage 1 MSE trajectory panels
    4. w1_traj_panels_<axis>.png      — per-config W1 trajectory panels
    5. w2_traj_panels_<axis>.png      — per-config W2 trajectory panels
    6. all_mse_traj_<axis>.png/svg    — overlaid mean MSE trajectories
    7. all_w1_traj_<axis>.png/svg     — overlaid mean W1 trajectories
    8. all_w2_traj_<axis>.png/svg     — overlaid mean W2 trajectories

Edit the settings block in main() or the CURVE_STYLES dict below.

Usage:
    python fig4_mse_sweep_replot.py --sweep N
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from fig3_mixture_prior import mixture_log_density

DIR_OUT    = os.path.join(SCRIPT_DIR, 'fig4_sweep_output')

# ---------------------------------------------------------------------------
# Curve style globals — edit here to change appearance without touching plots
# Each entry: (npz_metric_key, linestyle, color, linewidth, label)
# These are used in Figure 1 (scalar MSE vs sweep axis) and trajectory panels.
# ---------------------------------------------------------------------------
CURVE_STYLES = [
    # key                   ls     color           lw    label
    ('mse_init',           '--',  'gray',          1.5,  'noisy input  (l=0)'),
    ('mse_tweedie',        '-.',  'mediumorchid',  1.8,  'Stage 2 only (no prior refinement)'),
    ('mse_s1_opt',         '-',   'steelblue',     2.0,  'best Stage 1  (opt depth)'),
    ('mse_s2_opt',         '-',   'darkorchid',    2.0,  'best Stage 2  (opt depth)'),
    ('mse_s2',             '--',  'crimson',        1.5,  'Stage 2 final  (l=L)'),
    ('bayes_mse',          ':',   '#c49a6c',       1.8,  'Bayes MMSE  (finite-N)'),
    ('analytic_bayes_mse', ':',   'black',         1.8,  'Bayes MMSE  (analytic)'),
]

# Band alpha for ±1σ shading (set 0 to disable per-curve)
BAND_ALPHA = {
    'mse_init':           0.12,
    'mse_tweedie':        0.12,
    'mse_s1_opt':         0.15,
    'mse_s2_opt':         0.15,
    'mse_s2':             0.10,
    'bayes_mse':          0.10,
    'analytic_bayes_mse': 0.0,   # scalar — no band
}

# Horizontal reference lines on trajectory panels (key, color, ls, label)
TRAJ_HREFS = [
    ('analytic_bayes_mse', 'black',        ':',  'Bayes (analytic)'),
    ('bayes_mse',          '#c49a6c',      ':',  'Bayes (finite-N, mean)'),
    ('mse_tweedie',        'mediumorchid', '-.', 'Stage 2 only (no prior refinement, mean)'),
    ('mse_s2_opt',         'darkorchid',  '-',  'best Stage 2 (mean)'),
    ('mse_s2',             'crimson',     '--', 'Stage 2 final l=L (mean)'),
]

# W1/W2 plot color
W_COLOR = 'steelblue'


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _val_key(val):
    """Consistent npz key prefix — must match fig4_mse_sweep_run.py."""
    v = int(val) if float(val) == int(val) else val
    return f'v{v}'.replace('.', 'p')


def sweep_colors(n, cmap_name='RdBu', lo=0.05, hi=0.95):
    cmap = plt.get_cmap(cmap_name)
    return [cmap(lo + (hi - lo) * (i / max(n - 1, 1))) for i in range(n)]


def _savefig(fig, fname, exts=('png',)):
    for ext in exts:
        path = os.path.join(DIR_OUT, f'{fname}.{ext}')
        fig.savefig(path, bbox_inches='tight', dpi=150)
        print(f"  Saved {path}")
    plt.close(fig)


def _get_metric(r, key):
    """
    Return (values_array, is_scalar) for a metric key.
    Scalar keys (analytic_bayes_mse) return a 0-d float; others return (n_repeat,) arrays.
    """
    v = r[key]
    if np.ndim(v) == 0:
        return float(v), True
    return np.asarray(v), False


# deepskyblue-to-white colormap for 2D histograms
_HIST_CMAP = LinearSegmentedColormap.from_list(
    'dsb_white', ['white', 'deepskyblue'], N=256)


def _plot_cloud_hist_panel(ax, cloud, pi, mu_list, a_list,
                           xlim, ylim, G=80, bins=60):
    """
    Draw one panel: 2D histogram of cloud particles in deepskyblue,
    overlaid with clean prior contours in dark green.
    """
    '''
    ax.hist2d(cloud[:, 0], cloud[:, 1],
              bins=bins, range=[xlim, ylim],
              cmap=_HIST_CMAP, density=True, zorder=1)'''
    idx = np.random.choice(len(cloud), size=min(2000, len(cloud)), replace=False)
    ax.scatter(cloud[idx, 0], cloud[idx, 1],
               s=3, alpha=0.25, color='deepskyblue', zorder=1)


    gx = np.linspace(xlim[0], xlim[1], G)
    gy = np.linspace(ylim[0], ylim[1], G)
    Gx, Gy = np.meshgrid(gx, gy)
    pts = np.stack([Gx, Gy], axis=-1)
    Z   = np.exp(mixture_log_density(pts, pi, mu_list, a_list, sigma2=0.0))
    ax.contour(Gx, Gy, Z, levels=5, colors='darkgreen',
               linewidths=0.7, alpha=0.55, zorder=2)

    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])


def plot_cloud_snapshot_grid(data, results, sweep_values, snap_cols,
                              base_title, sweep_axis_str, fname):
    """
    Grid of 2D histograms: rows = sweep values, columns = depth snapshots.

    snap_cols: list of (label_str, key_suffix) where key_suffix is one of:
        'z0'         — noisy input  (uses z0_all)
        'stage2'     — Stage 2 output (uses snap_stage2)
        's1_best'    — best Stage 1 cloud (uses snap_s1_best)
        's2_best'    — best early-S2 cloud (uses snap_s2_best)
        int          — index into snap_depths (uses z_snaps_all[:, idx])

    Clouds are averaged across seeds (all n_repeat particles pooled).
    """
    n_rows = len(sweep_values)
    n_cols = len(snap_cols)

    # Determine axis limits from first config (shared across all panels)
    k0     = _val_key(sweep_values[0])
    z0_ref = data[f'{k0}_z0_all']          # (n_repeat, N, 2)
    pad    = 0.5
    all_xy = z0_ref.reshape(-1, 2)
    xlim   = (float(all_xy[:, 0].min()) - pad, float(all_xy[:, 0].max()) + pad)
    ylim   = (float(all_xy[:, 1].min()) - pad, float(all_xy[:, 1].max()) + pad)

    # Mixture params — reconstruct per-config from stored 'a' scalar.
    # mu_list assumed symmetric ±1 along x-axis (fig4 default); contours
    # are indicative so this is fine even when sweeping 'a'.
    pi      = np.array([0.5, 0.5])
    mu_list = np.array([[-1.0, 0.0], [1.0, 0.0]])

    panel_w = 2.2
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(panel_w * n_cols, panel_w * n_rows),
                             squeeze=False)

    for row, val in enumerate(sweep_values):
        k     = _val_key(val)
        r     = results[val]
        z0_all       = data[f'{k}_z0_all']        # (n_repeat, N, 2)
        z_snaps_all  = data[f'{k}_z_snaps_all']   # (n_repeat, n_snaps, N, 2)
        snap_depths  = data[f'{k}_snap_depths']   # (n_snaps,)
        snap_s1_best = data[f'{k}_snap_s1_best']  # (n_repeat, N, 2)
        snap_stage2  = data[f'{k}_snap_stage2']   # (n_repeat, N, 2)
        snap_s2_best = data[f'{k}_snap_s2_best'] if f'{k}_snap_s2_best' in data \
                       else None

        # a_list for this config (may vary if sweep_axis == 'a')
        a_val_r  = float(r['a'])
        a_list_r = np.array([a_val_r, a_val_r])

        for col, (col_label, key_suf) in enumerate(snap_cols):
            ax = axes[row][col]

            if key_suf == 'z0':
                cloud = z0_all.reshape(-1, 2)
            elif key_suf == 'stage2':
                cloud = snap_stage2.reshape(-1, 2)
            elif key_suf == 's1_best':
                cloud = snap_s1_best.reshape(-1, 2)
            elif key_suf == 's2_best' and snap_s2_best is not None:
                cloud = snap_s2_best.reshape(-1, 2)
            elif isinstance(key_suf, int):
                # index into snap_depths — pick nearest
                idx = int(np.argmin(np.abs(snap_depths - key_suf)))
                cloud = z_snaps_all[:, idx, :, :].reshape(-1, 2)
            else:
                cloud = z0_all.reshape(-1, 2)  # fallback

            _plot_cloud_hist_panel(ax, cloud, pi, mu_list, a_list_r,
                                   xlim, ylim)

            if row == 0:
                ax.set_title(col_label, fontsize=8, pad=3)
            if col == 0:
                ax.set_ylabel(r['label'], fontsize=7, labelpad=3)

    fig.suptitle(f'Cloud snapshots  ({base_title})\ndeepskyblue = z histogram  |  green = clean prior',
                 fontsize=9, y=1.01)
    fig.tight_layout()
    _savefig(fig, fname, exts=('png',))
    #_savefig(fig, fname, exts=('pdf',))


# ---------------------------------------------------------------------------
# main
# ------------------------------ ---------------------------------------------

def main(sweep_axis='beta'):
    # =========================================================================
    # Settings — edit here
    # =========================================================================
    run_label   = f'sweep_{sweep_axis}'
    npz_path    = os.path.join(DIR_OUT, f'results_{run_label}.npz')

    show_bands  = True      # shade ±1 std band on trajectory overlay plots
    show_seeds  = True      # show per-seed dots on scalar summary plots
    seed_alpha  = 0.35      # alpha for per-seed dots
    cmap_name   = 'RdBu'    # colormap for sweep values (trajectory overlay)
    val_omit    = []        # sweep values to skip ([] = include all)

    # Extra S2-at-ℓ baselines: recompute Stage-2 MSE using the stored cloud
    # snapshot nearest to each requested depth.  [] = no extras.
    extra_s2_depths = [200]

    # Normalised MSE variants — produce extra copies of mse_vs_X and mse_traj_panels_X:
    #   'div_sigma2'  : MSE / σ²
    #   'sub_bayes'   : MSE − MSE_Bayes
    #   'div_bayes'   : MSE / MSE_Bayes
    # For sub_bayes / div_bayes, bayes_ref controls which Bayes MSE is used as denominator:
    #   'finite'   → mean of bayes_mse_runs  (finite-N, seed-dependent)
    #   'analytic' → analytic_bayes_mse      (20k-sample, approx ~deterministic)
    mse_norm_variants = ['div_sigma2', 'sub_bayes', 'div_bayes']
    bayes_ref         = 'finite'   # 'finite' | 'analytic'
    # =========================================================================

    print(f"Loading {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    sweep_values_all = data['sweep_values'].tolist()
    sweep_values     = [v for v in sweep_values_all if v not in val_omit]
    n_repeat         = int(data['n_repeat'])
    sweep_axis_str   = str(data['sweep_axis'])

    print(f"  sweep_axis={sweep_axis_str}  values={sweep_values}  n_repeat={n_repeat}")

    # Unpack per-config results
    results = {}
    for val in sweep_values:
        k = _val_key(val)
        d = dict(
            label=str(data[f'{k}_label']),
            beta=float(data[f'{k}_beta']),
            sigma2=float(data[f'{k}_sigma2']),
            N=int(data[f'{k}_N']),
            a=float(data[f'{k}_a']),
            eta=float(data[f'{k}_eta']),
            L=int(data[f'{k}_L']),
            T_run=float(data[f'{k}_T_run']),
            mse_s1_opt=data[f'{k}_mse_s1_opt'],

            mse_s2=data[f'{k}_mse_s2'],
            bayes_mse=data[f'{k}_bayes_mse'],
            mse_init=data[f'{k}_mse_init'],
            w1_s2=data[f'{k}_w1_s2'],
            w2_s2=data[f'{k}_w2_s2'],
            traj_indices=data[f'{k}_traj_indices'],
            mse_traj_mean=data[f'{k}_mse_traj_mean'],
            mse_traj_std=data[f'{k}_mse_traj_std'],
            w1_traj_mean=data[f'{k}_w1_traj_mean'],
            w1_traj_std=data[f'{k}_w1_traj_std'],
            w2_traj_mean=data[f'{k}_w2_traj_mean'],
            w2_traj_std=data[f'{k}_w2_traj_std'],
        )
        if f'{k}_early_s2_traj_mean' in data:
            d['early_s2_traj_mean'] = data[f'{k}_early_s2_traj_mean']
            d['early_s2_traj_std']  = data[f'{k}_early_s2_traj_std']
        if f'{k}_opt_s1_depth' in data:
            d['opt_s1_depth'] = data[f'{k}_opt_s1_depth']
            d['opt_s2_depth'] = data[f'{k}_opt_s2_depth']
        elif f'{k}_opt_depth' in data:  # legacy key
            d['opt_s1_depth'] = data[f'{k}_opt_depth']
        # New baselines (may be absent in older npz files)
        if f'{k}_mse_s2_opt' in data:
            d['mse_s2_opt'] = data[f'{k}_mse_s2_opt']
        if f'{k}_mse_tweedie' in data:
            d['mse_tweedie'] = data[f'{k}_mse_tweedie']
        if f'{k}_analytic_bayes_mse' in data:
            d['analytic_bayes_mse'] = data[f'{k}_analytic_bayes_mse']
        if f'{k}_w1_full_s2' in data:
            d['w1_full_s2'] = data[f'{k}_w1_full_s2']
            d['w2_full_s2'] = data[f'{k}_w2_full_s2']
        results[val] = d
        print(f"  Loaded val={val}  beta={d['beta']:.3g}  sigma2={d['sigma2']:.3g}  N={d['N']}  a={d['a']:.3g}  L={d['L']}  T_run={d['T_run']:.1f}s")

    # -------------------------------------------------------------------------
    # Fallback: recompute mse_tweedie from stored z0_all if absent from npz.
    # Requires z0_all (shape (n_repeat, N, 2)) and config params — all present
    # in npz files produced by the current sweep_run.py.
    # -------------------------------------------------------------------------
    first = results[sweep_values[0]]
    if 'mse_tweedie' not in first and 'z0_all' not in first:
        print("  [replot] mse_tweedie absent and z0_all not stored — skipping Stage 2 only baseline")
    elif 'mse_tweedie' not in first:
        print("  [replot] Stage 2 only baseline absent from npz — recomputing from stored z0_all ...")
        from fig3_mixture_prior import gauss_cross_attn_nd, compute_mse_nd
        for val in sweep_values:
            k  = _val_key(val)
            r  = results[val]
            z0_all      = data[f'{k}_z0_all']       # (n_repeat, N, 2)
            x_clean_all = data[f'{k}_x_clean_all']  # (n_repeat, N, 2)
            beta_final  = 1.0 / r['sigma2']
            mse_tw = np.empty(n_repeat)
            for ri in range(n_repeat):
                z0_r      = z0_all[ri]
                tw_out    = gauss_cross_attn_nd(z0_r, z0_r, beta_final)
                mse_tw[ri] = compute_mse_nd(tw_out, x_clean_all[ri])
            r['mse_tweedie'] = mse_tw
        print("  [replot] Stage 2 only baseline recompute done.")

    # -------------------------------------------------------------------------
    # Extra S2-at-ℓ baselines: compute from stored z_snaps_all snapshots.
    # For each requested depth we find the nearest stored snap_depth and apply
    # Stage-2 cross-attention from z0 to that cloud.  A warning is printed when
    # the nearest snap differs from the requested depth.
    # -------------------------------------------------------------------------
    extra_s2_metric_keys = []   # populated below; used to extend CURVE_STYLES / TRAJ_HREFS
    if extra_s2_depths:
        from fig3_mixture_prior import gauss_cross_attn_nd, compute_mse_nd
        for req_l in extra_s2_depths:
            metric_key = f'mse_s2_at_l{req_l}'
            extra_s2_metric_keys.append((req_l, metric_key))
            for val in sweep_values:
                k  = _val_key(val)
                r  = results[val]
                if f'{k}_z_snaps_all' not in data:
                    print(f"  [extra_s2_depths] z_snaps_all not in npz for val={val} — skipping")
                    break
                z0_all_v    = data[f'{k}_z0_all']       # (n_repeat, N, 2)
                x_clean_all = data[f'{k}_x_clean_all']  # (n_repeat, N, 2)
                z_snaps     = data[f'{k}_z_snaps_all']  # (n_repeat, n_snaps, N, 2)
                snap_depths_v = data[f'{k}_snap_depths']  # (n_snaps,)
                beta_final  = 1.0 / r['sigma2']

                # find nearest stored snapshot
                nearest_idx = int(np.argmin(np.abs(snap_depths_v - req_l)))
                actual_l    = int(snap_depths_v[nearest_idx])
                if actual_l != req_l:
                    print(f"  [extra_s2_depths] val={val}: requested ℓ={req_l}, "
                          f"using nearest stored snapshot ℓ={actual_l}")

                mse_runs = np.empty(n_repeat)
                for ri in range(n_repeat):
                    z0_r     = z0_all_v[ri]                       # (N, 2)
                    cloud_r  = z_snaps[ri, nearest_idx]           # (N, 2)
                    s2_out   = gauss_cross_attn_nd(z0_r, cloud_r, beta_final)
                    mse_runs[ri] = compute_mse_nd(s2_out, x_clean_all[ri])
                r[metric_key] = mse_runs
                r[f'{metric_key}_actual_l'] = actual_l
            else:
                print(f"  [extra_s2_depths] computed {metric_key} "
                      f"(actual ℓ={results[sweep_values[0]].get(f'{metric_key}_actual_l', req_l)})"
                      f" for all {len(sweep_values)} configs")

    # Dynamically extend CURVE_STYLES and TRAJ_HREFS with extra baselines
    _extra_colors  = ['#2ca02c', '#8c564b', '#e377c2', '#bcbd22', '#17becf']
    active_curve_styles = list(CURVE_STYLES)
    active_traj_hrefs   = list(TRAJ_HREFS)
    for i, (req_l, mkey) in enumerate(extra_s2_metric_keys):
        color = _extra_colors[i % len(_extra_colors)]
        actual_l = results[sweep_values[0]].get(f'{mkey}_actual_l', req_l)
        lbl = f'S2 at ℓ={actual_l}' + (f'  [req {req_l}]' if actual_l != req_l else '')
        active_curve_styles.append((mkey, ':', color, 1.8, lbl))
        active_traj_hrefs.append((mkey, color, ':', lbl))

    # Only include curve styles whose key is present in the data
    active_styles = [s for s in active_curve_styles if s[0] in results[sweep_values[0]]]

    colors = sweep_colors(len(sweep_values), cmap_name=cmap_name)

    sweep_xlabel = {
        'beta':   r'Attention bandwidth  $\beta$',
        'sigma2': r'Noise variance  $\sigma^2$',
        'N':      'Particle count  $N$',
        'a':      'Component std dev  $a$',
    }.get(sweep_axis_str, sweep_axis_str)

    base_title = f'{sweep_axis_str} sweep  |  n_repeat={n_repeat}  [{run_label}]'

    x_vals = np.array(sweep_values)

    # -------------------------------------------------------------------------
    # Figure 1: scalar MSE summary vs sweep axis
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for metric_key, ls, color, lw, zlabel in active_styles:
        vals_per_config = [_get_metric(results[v], metric_key) for v in sweep_values]
        means, stds = [], []
        for a, s in vals_per_config:
            if s:
                means.append(float(a)); stds.append(0.0)
            else:
                means.append(a.mean()); stds.append(a.std())
        means = np.array(means)
        stds  = np.array(stds)

        band_a = BAND_ALPHA.get(metric_key, 0.12)
        if band_a > 0 and stds.max() > 0:
            ax.fill_between(x_vals, means - stds, means + stds,
                            color=color, alpha=band_a)
        ax.plot(x_vals, means, ls, color=color, linewidth=lw, label=zlabel)

        if show_seeds:
            for v, (a, s) in zip(sweep_values, vals_per_config):
                if not s:
                    ax.scatter(np.full(n_repeat, v), a,
                               s=12, color=color, alpha=seed_alpha, zorder=5)

    ax.set_xlabel(sweep_xlabel)
    ax.set_ylabel('MSE')
    ax.set_title(f'MSE vs {sweep_axis_str}  (mean ± 1σ over {n_repeat} seeds)\n{base_title}')
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    _savefig(fig, f'mse_vs_{sweep_axis_str}')

    # -------------------------------------------------------------------------
    # Figure 1 variants: normalised MSE vs sweep axis
    # -------------------------------------------------------------------------
    def _bayes_ref_per_config(r):
        """Return the per-config Bayes MSE scalar used as reference."""
        if bayes_ref == 'analytic' and 'analytic_bayes_mse' in r:
            return float(r['analytic_bayes_mse'])
        if 'bayes_mse' in r:
            return float(np.asarray(r['bayes_mse']).mean())
        return None

    norm_spec = {
        'div_sigma2': dict(
            fname_suffix='div_sigma2',
            ylabel=r'MSE / $\sigma^2$',
            title_tag=r'MSE / $\sigma^2$',
            transform=lambda means, stds, r: (
                means / r['sigma2'],
                stds  / r['sigma2'],
            ),
            href_transform=lambda hval, r: hval / r['sigma2'],
        ),
        'sub_bayes': dict(
            fname_suffix='sub_bayes',
            ylabel=r'MSE $-$ MSE$_\mathrm{Bayes}$',
            title_tag=r'MSE − MSE_Bayes',
            transform=lambda means, stds, r: (
                means - _bayes_ref_per_config(r),
                stds,
            ),
            href_transform=lambda hval, r: hval - _bayes_ref_per_config(r),
        ),
        'div_bayes': dict(
            fname_suffix='div_bayes',
            ylabel=r'MSE / MSE$_\mathrm{Bayes}$',
            title_tag=r'MSE / MSE_Bayes',
            transform=lambda means, stds, r: (
                means / _bayes_ref_per_config(r),
                stds  / _bayes_ref_per_config(r),
            ),
            href_transform=lambda hval, r: hval / _bayes_ref_per_config(r),
        ),
    }

    for norm_key in mse_norm_variants:
        if norm_key not in norm_spec:
            continue
        sp = norm_spec[norm_key]
        # skip if bayes ref is unavailable
        if norm_key in ('sub_bayes', 'div_bayes'):
            if _bayes_ref_per_config(results[sweep_values[0]]) is None:
                print(f"  [skip norm variant {norm_key}] bayes ref not available")
                continue

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for metric_key, ls, color, lw, zlabel in active_styles:
            vals_per_config = [_get_metric(results[v], metric_key) for v in sweep_values]
            t_means, t_stds = [], []
            for (a, s), v in zip(vals_per_config, sweep_values):
                r = results[v]
                raw_mean = float(a)        if s else a.mean()
                raw_std  = 0.0             if s else a.std()
                tm, ts   = sp['transform'](np.array([raw_mean]), np.array([raw_std]), r)
                t_means.append(float(tm[0])); t_stds.append(float(ts[0]))
            t_means = np.array(t_means)
            t_stds  = np.array(t_stds)

            band_a = BAND_ALPHA.get(metric_key, 0.12)
            if band_a > 0 and t_stds.max() > 0:
                ax.fill_between(x_vals, t_means - t_stds, t_means + t_stds,
                                color=color, alpha=band_a)
            ax.plot(x_vals, t_means, ls, color=color, linewidth=lw, label=zlabel)

            if show_seeds:
                for v, (a, s) in zip(sweep_values, vals_per_config):
                    if not s:
                        r = results[v]
                        t_a, _ = sp['transform'](a, np.zeros_like(a), r)
                        ax.scatter(np.full(n_repeat, v), t_a,
                                   s=12, color=color, alpha=seed_alpha, zorder=5)

        ax.set_xlabel(sweep_xlabel)
        ax.set_ylabel(sp['ylabel'])
        ax.set_title(f'{sp["title_tag"]} vs {sweep_axis_str}  '
                     f'(mean ± 1σ over {n_repeat} seeds)\n{base_title}')
        ax.legend(fontsize=8)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        _savefig(fig, f'mse_vs_{sweep_axis_str}_{sp["fname_suffix"]}')

    # -------------------------------------------------------------------------
    # Figure 2: W1 (sliced) / W2 at Stage 2 vs sweep axis
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, metric_key, ylabel in zip(
            axes,
            ['w1_s2', 'w2_s2'],
            ['W1 (sliced) at Stage 2', 'W2 (exact, subsampled) at Stage 2']):
        means = np.array([results[v][metric_key].mean() for v in sweep_values])
        stds  = np.array([results[v][metric_key].std()  for v in sweep_values])
        ax.fill_between(x_vals, means - stds, means + stds,
                        color=W_COLOR, alpha=0.18)
        ax.plot(x_vals, means, '-', color=W_COLOR, linewidth=2.0,
                label='sliced' if 'w1' in metric_key else 'exact (subsampled)')

        if show_seeds:
            for v in sweep_values:
                ax.scatter(np.full(n_repeat, v), results[v][metric_key],
                           s=12, color=W_COLOR, alpha=seed_alpha, zorder=5)

        if 'w1_full_s2' in results[sweep_values[0]]:
            fkey   = 'w1_full_s2' if 'w1' in metric_key else 'w2_full_s2'
            fmeans = np.array([results[v][fkey].mean() for v in sweep_values])
            ax.plot(x_vals, fmeans, '--', color='navy', linewidth=1.5,
                    label='exact (full LP)')
        ax.legend(fontsize=8)
        ax.set_xlabel(sweep_xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel}\n{base_title}')
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    _savefig(fig, f'w1w2_vs_{sweep_axis_str}')

    # -------------------------------------------------------------------------
    # Figures 3–5: per-config trajectory panels (MSE, W1, W2)
    # -------------------------------------------------------------------------
    active_hrefs = [h for h in active_traj_hrefs if h[0] in results[sweep_values[0]]]

    # include early_s2 trajectory panels only if present in data
    traj_panel_list = [
        ('mse',      'MSE',              'mse_traj'),
        ('w1',       'W1 (sliced)',      'w1_traj'),
        ('w2',       'W2',               'w2_traj'),
    ]
    if 'early_s2_traj_mean' in results[sweep_values[0]]:
        traj_panel_list.append(
            ('early_s2', 'MSE early S2 (cross-β)', 'early_s2_traj'))

    for traj_key, ylabel, fname_prefix in traj_panel_list:
        traj_mean_key = f'{traj_key}_traj_mean'
        traj_std_key  = f'{traj_key}_traj_std'
        fig, axes = plt.subplots(1, len(sweep_values),
                                 figsize=(3.5 * len(sweep_values), 3.5),
                                 sharey=False, squeeze=False)
        for col, val in enumerate(sweep_values):
            ax = axes[0][col]
            r  = results[val]
            t_idx = r['traj_indices']
            mean  = r[traj_mean_key]
            std   = r[traj_std_key]

            ax.plot(t_idx, mean, '-', color='steelblue', linewidth=1.8)
            if show_bands:
                ax.fill_between(t_idx, mean - std, mean + std,
                                color='steelblue', alpha=0.20)

            if traj_key in ('mse', 'early_s2'):
                for href_key, href_color, href_ls, href_label in active_hrefs:
                    a, s = _get_metric(r, href_key)
                    hval = float(a) if s else float(a.mean())
                    ax.axhline(hval, color=href_color, linestyle=href_ls,
                               linewidth=1.2, label=href_label)
                ax.legend(fontsize=5)

            ax.set_title(r['label'], fontsize=8)
            ax.set_xlabel('Layer $\\ell$', fontsize=7)
            ax.set_ylabel(ylabel if col == 0 else '', fontsize=7)
            ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.4)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        fig.suptitle(f'{ylabel} trajectory  ({base_title})', fontsize=9)
        fig.tight_layout()
        _savefig(fig, f'{fname_prefix}_panels_{sweep_axis_str}', exts=('png',))

    # -------------------------------------------------------------------------
    # Normalised MSE trajectory panels (mse and early_s2 only)
    # -------------------------------------------------------------------------
    mse_traj_norm_keys = ['mse']
    if 'early_s2_traj_mean' in results[sweep_values[0]]:
        mse_traj_norm_keys.append('early_s2')

    for norm_key in mse_norm_variants:
        if norm_key not in norm_spec:
            continue
        if norm_key in ('sub_bayes', 'div_bayes'):
            if _bayes_ref_per_config(results[sweep_values[0]]) is None:
                continue
        sp = norm_spec[norm_key]

        for traj_key in mse_traj_norm_keys:
            traj_mean_key = f'{traj_key}_traj_mean'
            traj_std_key  = f'{traj_key}_traj_std'
            ylabel_base   = 'MSE' if traj_key == 'mse' else 'MSE early S2'

            fig, axes = plt.subplots(1, len(sweep_values),
                                     figsize=(3.5 * len(sweep_values), 3.5),
                                     sharey=False, squeeze=False)
            for col, val in enumerate(sweep_values):
                ax = axes[0][col]
                r  = results[val]
                t_idx = r['traj_indices']
                raw_mean = r[traj_mean_key]
                raw_std  = r[traj_std_key]
                mean, std = sp['transform'](raw_mean, raw_std, r)

                ax.plot(t_idx, mean, '-', color='steelblue', linewidth=1.8)
                if show_bands:
                    ax.fill_between(t_idx, mean - std, mean + std,
                                    color='steelblue', alpha=0.20)

                # horizontal refs — same transform applied
                for href_key, href_color, href_ls, href_label in active_hrefs:
                    if href_key not in r:
                        continue
                    a, s = _get_metric(r, href_key)
                    hval = float(a) if s else float(a.mean())
                    t_hval = sp['href_transform'](hval, r)
                    ax.axhline(t_hval, color=href_color, linestyle=href_ls,
                               linewidth=1.2, label=href_label)
                ax.legend(fontsize=5)

                ax.set_title(r['label'], fontsize=8)
                ax.set_xlabel('Layer $\\ell$', fontsize=7)
                ax.set_ylabel(sp['ylabel'] if col == 0 else '', fontsize=7)
                ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.4)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            fig.suptitle(f'{ylabel_base} ({sp["title_tag"]}) trajectory  ({base_title})',
                         fontsize=9)
            fig.tight_layout()
            _savefig(fig, f'{traj_key}_traj_panels_{sweep_axis_str}_{sp["fname_suffix"]}',
                     exts=('png',))

    # -------------------------------------------------------------------------
    # Figures 6–8: overlay of mean trajectories, one curve per val
    # -------------------------------------------------------------------------
    traj_overlay_list = [
        ('mse',      'MSE',                      'all_mse_traj'),
        ('w1',       'W1 (sliced)',               'all_w1_traj'),
        ('w2',       'W2',                        'all_w2_traj'),
    ]
    if 'early_s2_traj_mean' in results[sweep_values[0]]:
        traj_overlay_list.append(
            ('early_s2', 'MSE early S2 (cross-β)', 'all_early_s2_traj'))

    for traj_key, ylabel, fname_prefix in traj_overlay_list:
        traj_mean_key = f'{traj_key}_traj_mean'
        traj_std_key  = f'{traj_key}_traj_std'
        fig, ax = plt.subplots(figsize=(7, 4))
        for color, val in zip(colors, sweep_values):
            r     = results[val]
            t_idx = r['traj_indices']
            mean  = r[traj_mean_key]
            std   = r[traj_std_key]
            ax.plot(t_idx, mean, '-', color=color, linewidth=1.8,
                    label=r['label'])
            if show_bands:
                ax.fill_between(t_idx, mean - std, mean + std,
                                color=color, alpha=0.12)

        ax.set_xlabel('Layer $\\ell$')
        ax.set_ylabel(f'{ylabel}  [mean over seeds]')
        ax.set_title(f'{ylabel} trajectory vs {sweep_axis_str}  '
                     f'(mean ± 1σ over {n_repeat} seeds)\n{base_title}')
        ax.legend(fontsize=8)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        _savefig(fig, f'{fname_prefix}_{sweep_axis_str}')

    # -------------------------------------------------------------------------
    # Figure: combined Stage 1 + Stage 2 trajectories with baselines
    # One colored pair of curves per sweep value:
    #   solid  = Stage 1 cloud MSE  (mse_traj)
    #   dashed = early Stage 2 MSE  (early_s2_traj, cross-beta), i.e. what Stage 2
    #            achieves if applied at each depth — final dot = Stage 2 at l=L
    # Horizontal baselines drawn once in neutral colors (Bayes, Stage 2 only, etc.)
    # -------------------------------------------------------------------------
    has_early_s2 = 'early_s2_traj_mean' in results[sweep_values[0]]
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Horizontal baselines (drawn first, behind trajectories)
    baseline_drawn = set()
    for val in sweep_values:
        r = results[val]
        for href_key, href_color, href_ls, href_label in active_hrefs:
            if href_key not in r or href_key in baseline_drawn:
                continue
            a, s = _get_metric(r, href_key)
            hval = float(a) if s else float(a.mean())
            ax.axhline(hval, color=href_color, linestyle=href_ls,
                       linewidth=1.3, alpha=0.85, label=href_label, zorder=1)
            baseline_drawn.add(href_key)

    # Trajectory curves: one color per sweep value
    s1_handle = s2_handle = None
    for color, val in zip(colors, sweep_values):
        r     = results[val]
        t_idx = r['traj_indices']
        s1_mean = r['mse_traj_mean']
        s1_std  = r['mse_traj_std']

        l_s1, = ax.plot(t_idx, s1_mean, '-', color=color, linewidth=1.8,
                        label=r['label'], zorder=3)
        if show_bands:
            ax.fill_between(t_idx, s1_mean - s1_std, s1_mean + s1_std,
                            color=color, alpha=0.10, zorder=2)
        if s1_handle is None:
            s1_handle = l_s1

        if has_early_s2:
            s2_mean = r['early_s2_traj_mean']
            s2_std  = r['early_s2_traj_std']
            l_s2, = ax.plot(t_idx, s2_mean, '--', color=color, linewidth=1.4,
                            zorder=3)
            if show_bands:
                ax.fill_between(t_idx, s2_mean - s2_std, s2_mean + s2_std,
                                color=color, alpha=0.07, zorder=2)
            if s2_handle is None:
                s2_handle = l_s2

        # Mark final Stage 2 scalar as a dot at l=L
        if 'mse_s2' in r:
            s2_final_mean = float(np.asarray(r['mse_s2']).mean())
            ax.scatter([t_idx[-1]], [s2_final_mean],
                       s=30, color=color, marker='D',
                       edgecolors='black', linewidths=0.5, zorder=5)

    # Legend: sweep value lines + style proxy entries
    handles, labels = ax.get_legend_handles_labels()
    # Add proxy lines explaining solid vs dashed
    import matplotlib.lines as mlines
    proxy_s1 = mlines.Line2D([], [], color='black', ls='-',  lw=1.5, label='Stage 1 cloud')
    proxy_s2 = mlines.Line2D([], [], color='black', ls='--', lw=1.5, label='Stage 2 (at each $\\ell$)')
    proxy_dot = mlines.Line2D([], [], color='black', ls='none', marker='D',
                               markersize=5, label='Stage 2 final ($\\ell=L$)')
    handles += [proxy_s1, proxy_s2, proxy_dot]
    labels  += ['Stage 1 cloud', 'Stage 2 (at each $\\ell$)', 'Stage 2 final ($\\ell=L$)']

    ax.legend(handles=handles, labels=labels, fontsize=7,
              ncol=2, loc='upper right')
    ax.set_xlabel('Layer  $\\ell$')
    ax.set_ylabel('MSE  [mean over seeds]')
    ax.set_title(f'Stage 1 + Stage 2 MSE vs depth  |  {sweep_axis_str} sweep\n'
                 f'solid=S1 cloud,  dashed=S2 at each ℓ,  ◆=S2 final\n{base_title}',
                 fontsize=9)
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    _savefig(fig, f'mse_s1s2_combined_{sweep_axis_str}')

    # -------------------------------------------------------------------------
    # Normalised variants of the combined S1+S2 trajectory plot
    # -------------------------------------------------------------------------
    for norm_key in mse_norm_variants:
        if norm_key not in norm_spec:
            continue
        if norm_key in ('sub_bayes', 'div_bayes'):
            if _bayes_ref_per_config(results[sweep_values[0]]) is None:
                print(f"  [skip combined norm variant {norm_key}] bayes ref not available")
                continue
        sp = norm_spec[norm_key]

        fig, ax = plt.subplots(figsize=(8, 4.5))

        # Horizontal baselines (same logic as raw, but transformed)
        baseline_drawn = set()
        for val in sweep_values:
            r = results[val]
            for href_key, href_color, href_ls, href_label in active_hrefs:
                if href_key not in r or href_key in baseline_drawn:
                    continue
                a, s = _get_metric(r, href_key)
                hval = float(a) if s else float(a.mean())
                t_hval = sp['href_transform'](hval, r)
                ax.axhline(t_hval, color=href_color, linestyle=href_ls,
                           linewidth=1.3, alpha=0.85, label=href_label, zorder=1)
                baseline_drawn.add(href_key)

        s1_handle = s2_handle = None
        for color, val in zip(colors, sweep_values):
            r     = results[val]
            t_idx = r['traj_indices']

            s1_mean_t, s1_std_t = sp['transform'](
                r['mse_traj_mean'], r['mse_traj_std'], r)
            l_s1, = ax.plot(t_idx, s1_mean_t, '-', color=color, linewidth=1.8,
                            label=r['label'], zorder=3)
            if show_bands:
                ax.fill_between(t_idx, s1_mean_t - s1_std_t, s1_mean_t + s1_std_t,
                                color=color, alpha=0.10, zorder=2)
            if s1_handle is None:
                s1_handle = l_s1

            if has_early_s2:
                s2_mean_t, s2_std_t = sp['transform'](
                    r['early_s2_traj_mean'], r['early_s2_traj_std'], r)
                l_s2, = ax.plot(t_idx, s2_mean_t, '--', color=color, linewidth=1.4,
                                zorder=3)
                if show_bands:
                    ax.fill_between(t_idx, s2_mean_t - s2_std_t, s2_mean_t + s2_std_t,
                                    color=color, alpha=0.07, zorder=2)
                if s2_handle is None:
                    s2_handle = l_s2

            if 'mse_s2' in r:
                s2_final_raw = float(np.asarray(r['mse_s2']).mean())
                s2_final_t, _ = sp['transform'](
                    np.array([s2_final_raw]), np.array([0.0]), r)
                ax.scatter([t_idx[-1]], [float(s2_final_t[0])],
                           s=30, color=color, marker='D',
                           edgecolors='black', linewidths=0.5, zorder=5)

        handles, labels = ax.get_legend_handles_labels()
        proxy_s1  = mlines.Line2D([], [], color='black', ls='-',    lw=1.5, label='Stage 1 cloud')
        proxy_s2  = mlines.Line2D([], [], color='black', ls='--',   lw=1.5, label='Stage 2 (at each $\\ell$)')
        proxy_dot = mlines.Line2D([], [], color='black', ls='none', marker='D',
                                  markersize=5, label='Stage 2 final ($\\ell=L$)')
        handles += [proxy_s1, proxy_s2, proxy_dot]
        labels  += ['Stage 1 cloud', 'Stage 2 (at each $\\ell$)', 'Stage 2 final ($\\ell=L$)']

        ax.legend(handles=handles, labels=labels, fontsize=7, ncol=2, loc='upper right')
        ax.set_xlabel('Layer  $\\ell$')
        ax.set_ylabel(f'{sp["ylabel"]}  [mean over seeds]')
        ax.set_title(
            f'Stage 1 + Stage 2 {sp["title_tag"]} vs depth  |  {sweep_axis_str} sweep\n'
            f'solid=S1 cloud,  dashed=S2 at each ℓ,  ◆=S2 final\n{base_title}',
            fontsize=9)
        ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        _savefig(fig, f'mse_s1s2_combined_{sweep_axis_str}_{sp["fname_suffix"]}')

    # -------------------------------------------------------------------------
    # Figure: cloud snapshot grid  (rows = sweep values, cols = depth snapshots)
    # -------------------------------------------------------------------------
    first_r = results[sweep_values[0]]
    if f'{_val_key(sweep_values[0])}_z_snaps_all' in data:
        snap_depths_0 = data[f'{_val_key(sweep_values[0])}_snap_depths']
        # pick ~4 evenly spaced depth snapshots plus stage2
        n_sd   = len(snap_depths_0)
        step   = max(1, n_sd // 4)
        depth_cols = [(f'$\\ell={int(snap_depths_0[i])}$', int(snap_depths_0[i]))
                      for i in range(0, n_sd, step)]
        snap_cols = (
            [('noisy input  $z^{(0)}$', 'z0')]
            + depth_cols
            + [('Stage 1 best', 's1_best'), ('Stage 2', 'stage2')]
        )
        plot_cloud_snapshot_grid(
            data, results, sweep_values,
            snap_cols=snap_cols,
            base_title=base_title,
            sweep_axis_str=sweep_axis_str,
            fname=f'cloud_snapshots_{sweep_axis_str}',
        )

    print(f"\nDone. Outputs in: {DIR_OUT}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Replot a saved Gaussian-mixture MSE sweep.')
    parser.add_argument('--sweep', choices=['N', 'sigma2', 'beta', 'a'], default='beta',
                        help='Sweep axis used by the simulation (default: beta).')
    main(sweep_axis=parser.parse_args().sweep)
