"""
Parameter sweep for the 2D Gaussian mixture denoising experiment (Expt 4).

Runs a grid of model configurations (primary sweep axis + fixed axes), with
multiple independent seeds per configuration.  For each seed the full Stage 1
trajectory is run and key metrics + cloud snapshots are stored.  Run
fig4_mse_sweep_replot.py to produce figures from the saved npz.

Sweep design
------------
Primary sweep axis  (choose one in the settings block):
    'beta'   — vary attention bandwidth
    'sigma2' — vary noise variance
    'N'      — vary particle count
    'a'      — vary component std dev (isotropic, both components equal)

For each configuration the same theory parameterization is used as in
fig3_mixture_prior.py:
    eta = beta * sigma2 / (2 * L_theory)
    L   = L_factor * L_theory

Sampling strategy (nested / Option B)
--------------------------------------
For the N sweep all seeds share the same underlying particles: we draw N_max
samples once per seed, then slice to the config's N.  This isolates the effect
of particle count from sample-identity variance — the N=500 run literally uses
a subset of the N=8000 particles.  For non-N sweeps a standard fresh draw of
N_base particles is used each seed.

Saved quantities per (config, seed)
-------------------------------------
Scalars (shape (n_repeat,)):
    mse_init, mse_s1_opt, mse_s2, mse_tweedie, bayes_mse, w1_s2, w2_s2,
    opt_s1_depth, opt_s2_depth

Near-deterministic scalars (single float, computed once per config):
    analytic_bayes_mse

Trajectory arrays (mean + std over seeds, shape (n_traj_pts,)):
    mse_traj, w1_traj, w2_traj, early_s2_traj  vs recorded layer indices

Cloud snapshots (shape (n_repeat, n_snaps, N, 2)):
    z0_snaps      — noisy inputs  (n_snaps=1 so shape (n_repeat, 1, N, 2))
    x_clean_snaps — clean targets (same)
    z_snaps       — Stage 1 cloud at snap_depths  (n_snaps depth points)
    snap_depths   — layer indices of the snapshots (shared across seeds)
    snap_s1_best  — cloud at opt_s1_depth, per seed  (n_repeat, N, 2)
    snap_s2_best  — cloud at opt_s2_depth (best early-S2), per seed  (n_repeat, N, 2)
    snap_stage2   — Stage 2 output, per seed  (n_repeat, N, 2)

Usage:
    python fig4_mse_sweep_run.py --sweep N
"""

import argparse
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from fig3_mixture_prior import (
    run_experiment,
    gauss_cross_attn_nd,
    compute_mse_nd,
    mixture_posterior_mean,
    sample_mixture,
    run_trajectory_nd,
    compute_metrics_trajectory_nd,
)

DIR_OUT = os.path.join(SCRIPT_DIR, 'fig4_sweep_output')
os.makedirs(DIR_OUT, exist_ok=True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _val_key(val):
    """Consistent npz key prefix for a sweep value (int-like values stay as int)."""
    v = int(val) if float(val) == int(val) else val
    return f'v{v}'.replace('.', 'p')


def _make_label(sweep_axis, val, cfg):
    """Short human-readable label for a sweep config value."""
    if sweep_axis == 'beta':
        return f'beta={val}'
    if sweep_axis == 'sigma2':
        return f'sigma2={val}'
    if sweep_axis == 'N':
        return f'N={val}'
    if sweep_axis == 'a':
        return f'a={val}'
    return f'{sweep_axis}={val}'


def _snap_layer_indices(L, every=20):
    """
    Layer indices to snapshot: 0, every, 2*every, ..., L (always inclusive).
    Returns sorted list of ints.
    """
    pts = list(range(0, L, every))
    if L not in pts:
        pts.append(L)
    return pts


def _nearest_recorded(target_depth, recorded_indices):
    """Return the element of recorded_indices closest to target_depth."""
    return min(recorded_indices, key=lambda d: abs(d - target_depth))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(sweep_axis='beta'):
    # =========================================================================
    # Settings — edit here
    # =========================================================================

    # --- Mixture prior (fixed across sweep) ---
    pi      = np.array([0.5, 0.5])
    mu_list = np.array([[-1.0, 0.0], [1.0, 0.0]])
    a_base  = 0.1          # default component std dev (used when 'a' is not sweep axis); a=0.4 std --> a^2=0.16 var

    # --- Noise (default; overridden per point when sweep_axis='sigma2') ---
    sigma2_base = 0.5

    # --- Dynamics (defaults) ---
    N_base      = 5000      # N=5000 default for figure
    beta_base   = 20.0
    L_theory    = 200       # reference depth; eta = beta*sigma2 / (2*L_theory); default: 200
    L_factor    = 3         # run to L_factor * L_theory layers; default 3 (e.g. L_max=600 for L_theory=200)

    # --- Sweep values (choose the axis with --sweep) ---

    sweep_values = {
        'beta':   [5, 10, 20, 30, 50, 100],
        'sigma2': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.75, 1.0],
        #'N':      [10, 50, 100, 200],
        'N':      [30, 50, 100, 200, 500, 1000, 1500, 2000, 3000, 5000, 8000],
        'a':      [0.2, 0.3, 0.5, 0.7, 1.0],
    }[sweep_axis]

    # --- Repeat settings ---
    n_repeat  = 8     # independent num. seeds per config
    seed_base = 0     # seed for repeat r is seed_base + r * 1000

    # --- Full Wasserstein (slow — set True only for small N or short L) ---
    run_full_wasserstein = False

    # --- Record trajectory every this many layers (controls memory/time) ---
    record_every = 5

    # --- Cloud snapshot cadence (layers) ---
    snap_every = 20   # snapshot every this many layers + always layer 0 and L

    # =========================================================================

    # N_max for nested sampling: largest N in sweep (or N_base if not N sweep)
    N_max = int(max(sweep_values)) if sweep_axis == 'N' else N_base

    run_label = f'sweep_{sweep_axis}'
    print(f"\n=== Fig 4 parameter sweep  ({run_label}) ===")
    print(f"  sweep_axis={sweep_axis}  values={sweep_values}")
    print(f"  n_repeat={n_repeat}  seed_base={seed_base}")
    print(f"  N_base={N_base}  N_max={N_max}  beta_base={beta_base}  sigma2_base={sigma2_base}")
    print(f"  L_theory={L_theory}  L_factor={L_factor}  record_every={record_every}")
    print(f"  snap_every={snap_every}  run_full_wasserstein={run_full_wasserstein}\n")
    t_total_start = time.perf_counter()

    results = {}   # val -> dict of arrays over seeds

    for val in sweep_values:
        # --- Resolve per-config parameters ---
        beta   = val          if sweep_axis == 'beta'   else beta_base
        sigma2 = val          if sweep_axis == 'sigma2' else sigma2_base
        N      = int(val)     if sweep_axis == 'N'      else N_base
        a      = float(val)   if sweep_axis == 'a'      else a_base

        a_list = np.array([a, a])

        # Theory parameterization
        eta  = beta * sigma2 / (2 * L_theory)
        if eta >= 1.0:
            L_theory_adj = int(np.ceil(beta * sigma2 / (2 * 0.9)))
            eta = beta * sigma2 / (2 * L_theory_adj)
            print(f"  [{_make_label(sweep_axis, val, None)}] clamped L_theory "
                  f"{L_theory} -> {L_theory_adj} to keep eta<1")
        else:
            L_theory_adj = L_theory
        L = L_factor * L_theory_adj

        beta_final_val = 1.0 / sigma2
        h      = eta / beta
        T_run  = L * h
        label  = _make_label(sweep_axis, val, None)

        # Snapshot layer indices: every snap_every layers, always 0 and L
        snap_depths = _snap_layer_indices(L, every=snap_every)
        n_snaps     = len(snap_depths)

        print(f"  {label}  eta={eta:.5f}  h={h:.6f}  L={L}  T_run={T_run:.4f}  "
              f"n_snaps={n_snaps}")
        t_config_start = time.perf_counter()

        # Analytic Bayes MSE: large-N draw so it's nearly deterministic across seeds.
        # Uses beta_final = 1/sigma2 as the Stage-2 bandwidth (same as run_experiment).
        rng_bayes = np.random.default_rng(999)
        x_large, _ = sample_mixture(pi, mu_list, a_list, 20000, rng_bayes)
        n_large     = x_large + rng_bayes.normal(0.0, np.sqrt(sigma2), size=x_large.shape)
        hat_bayes_large = mixture_posterior_mean(n_large, pi, mu_list, a_list, sigma2)
        analytic_bayes_mse = compute_mse_nd(hat_bayes_large, x_large)

        # Per-seed scalar arrays
        mse_s1_opt_runs  = np.empty(n_repeat)
        mse_s2_opt_runs  = np.empty(n_repeat)   # best early-S2 MSE (min over depth)
        mse_s2_runs      = np.empty(n_repeat)   # Stage 2 at l=L (final)
        bayes_mse_runs   = np.empty(n_repeat)
        mse_init_runs    = np.empty(n_repeat)
        mse_tweedie_runs = np.empty(n_repeat)
        w1_s2_runs       = np.empty(n_repeat)
        w2_s2_runs       = np.empty(n_repeat)
        opt_s1_depth_runs = np.empty(n_repeat, dtype=int)
        opt_s2_depth_runs = np.empty(n_repeat, dtype=int)
        w1_full_s2_runs  = np.empty(n_repeat) if run_full_wasserstein else None
        w2_full_s2_runs  = np.empty(n_repeat) if run_full_wasserstein else None

        # Per-seed trajectory lists
        traj_indices_ref   = None
        mse_traj_runs      = []
        w1_traj_runs       = []
        w2_traj_runs       = []
        early_s2_traj_runs = []   # early_s2_cross_beta trajectory

        # Per-seed cloud snapshot arrays  (n_repeat, n_snaps, N, 2)
        z0_all       = np.empty((n_repeat, N, 2))
        x_clean_all  = np.empty((n_repeat, N, 2))
        z_snaps_all  = np.empty((n_repeat, n_snaps, N, 2))
        snap_s1_best = np.empty((n_repeat, N, 2))   # cloud at opt S1 depth
        snap_s2_best = np.empty((n_repeat, N, 2))   # cloud at opt early-S2 depth
        snap_stage2  = np.empty((n_repeat, N, 2))   # Stage 2 output

        for r in range(n_repeat):
            seed = seed_base + r * 1000
            t_seed_start = time.perf_counter()

            # ------------------------------------------------------------------
            # Nested sampling (Option B):
            #   Draw N_max clean samples + noise once per seed.
            #   Slice to N for this config.  When sweep_axis != 'N', N == N_max
            #   so slicing is a no-op and behaviour is identical to Option A.
            #   This ensures that when varying N the smaller configs use a strict
            #   subset of the larger configs' particles, isolating N's effect.
            # ------------------------------------------------------------------
            rng_data = np.random.default_rng(seed)
            x_full, labels_full = sample_mixture(pi, mu_list, a_list, N_max, rng_data)
            noise_full = rng_data.normal(0.0, np.sqrt(sigma2), size=(N_max, 2))
            z0_full    = x_full + noise_full

            x_clean = x_full[:N]
            z0      = z0_full[:N]

            # Run dynamics directly (bypass run_experiment to reuse z0/x_clean)
            states, indices = run_trajectory_nd(
                z0, beta=beta, eta=eta, L=L,
                beta_final=beta_final_val,
                record_every=record_every,
            )

            rng_m = np.random.default_rng(seed + 1)
            metrics, bayes_mse = compute_metrics_trajectory_nd(
                states, x_clean, pi, mu_list, a_list, sigma2, beta,
                beta_cross=beta_final_val,
                N_sample=min(N, 2000), rng=rng_m,
                run_full_wasserstein=run_full_wasserstein,
            )

            t_seed = time.perf_counter() - t_seed_start
            print(f"    seed {r+1}/{n_repeat}  ({t_seed:.1f}s)")

            # Stage 2 only: apply the readout to the unrefined noisy context.
            tweedie_out = gauss_cross_attn_nd(z0, z0, beta_final_val)

            # Stage 1 best and early-S2 best depths
            s1_mask        = [i for i, d in enumerate(indices) if d < indices[-1]]
            recorded_s1    = [indices[i] for i in s1_mask]
            mse_s1         = [metrics['mse'][i]                   for i in s1_mask]
            early_s2_list  = [metrics['early_s2_cross_beta'][i]   for i in s1_mask]

            opt_s1_i = int(np.argmin(mse_s1))
            opt_s2_i = int(np.argmin(early_s2_list))

            mse_s1_opt_runs[r]   = mse_s1[opt_s1_i]
            mse_s2_opt_runs[r]   = early_s2_list[opt_s2_i]
            mse_s2_runs[r]       = metrics['mse'][-1]
            bayes_mse_runs[r]    = bayes_mse
            mse_init_runs[r]     = metrics['mse'][0]
            mse_tweedie_runs[r]  = compute_mse_nd(tweedie_out, x_clean)
            w1_s2_runs[r]        = metrics['w1'][-1]
            w2_s2_runs[r]        = metrics['w2'][-1]
            opt_s1_depth_runs[r] = indices[s1_mask[opt_s1_i]]
            opt_s2_depth_runs[r] = indices[s1_mask[opt_s2_i]]

            if run_full_wasserstein and 'w1_full' in metrics:
                w1_full_s2_runs[r] = metrics['w1_full'][-1]
                w2_full_s2_runs[r] = metrics['w2_full'][-1]

            # Metric trajectories (Stage 1 only)
            if traj_indices_ref is None:
                traj_indices_ref = recorded_s1
            mse_traj_runs.append(mse_s1)
            w1_traj_runs.append([metrics['w1'][i]              for i in s1_mask])
            w2_traj_runs.append([metrics['w2'][i]              for i in s1_mask])
            early_s2_traj_runs.append(early_s2_list)

            # Cloud snapshots at snap_depths
            idx_map = {indices[i]: i for i in range(len(indices))}
            z0_all[r]      = z0
            x_clean_all[r] = x_clean
            for s_i, sd in enumerate(snap_depths):
                nearest = _nearest_recorded(sd, recorded_s1)
                z_snaps_all[r, s_i] = states[idx_map[nearest]]

            snap_s1_best[r] = states[idx_map[_nearest_recorded(
                                  int(opt_s1_depth_runs[r]), recorded_s1)]]
            snap_s2_best[r] = states[idx_map[_nearest_recorded(
                                  int(opt_s2_depth_runs[r]), recorded_s1)]]
            snap_stage2[r]  = states[-1]   # Stage 2 output is last state

        mse_traj_arr      = np.array(mse_traj_runs)       # (n_repeat, T)
        w1_traj_arr       = np.array(w1_traj_runs)
        w2_traj_arr       = np.array(w2_traj_runs)
        early_s2_traj_arr = np.array(early_s2_traj_runs)

        t_config = time.perf_counter() - t_config_start
        print(f"  -> {label} done  total={t_config:.1f}s  "
              f"({t_config/n_repeat:.1f}s/seed)")

        results[val] = dict(
            label=label,
            beta=beta, sigma2=sigma2, N=N, a=a,
            eta=eta, L=L, T_run=T_run,
            analytic_bayes_mse=analytic_bayes_mse,
            mse_s1_opt_runs=mse_s1_opt_runs,
            mse_s2_opt_runs=mse_s2_opt_runs,
            mse_s2_runs=mse_s2_runs,
            bayes_mse_runs=bayes_mse_runs,
            mse_init_runs=mse_init_runs,
            mse_tweedie_runs=mse_tweedie_runs,
            w1_s2_runs=w1_s2_runs,
            w2_s2_runs=w2_s2_runs,
            opt_s1_depth_runs=opt_s1_depth_runs,
            opt_s2_depth_runs=opt_s2_depth_runs,
            w1_full_s2_runs=w1_full_s2_runs,
            w2_full_s2_runs=w2_full_s2_runs,
            traj_indices=np.array(traj_indices_ref),
            mse_traj_mean=mse_traj_arr.mean(axis=0),
            mse_traj_std=mse_traj_arr.std(axis=0),
            w1_traj_mean=w1_traj_arr.mean(axis=0),
            w1_traj_std=w1_traj_arr.std(axis=0),
            w2_traj_mean=w2_traj_arr.mean(axis=0),
            w2_traj_std=w2_traj_arr.std(axis=0),
            early_s2_traj_mean=early_s2_traj_arr.mean(axis=0),
            early_s2_traj_std=early_s2_traj_arr.std(axis=0),
            snap_depths=np.array(snap_depths),
            z0_all=z0_all,
            x_clean_all=x_clean_all,
            z_snaps_all=z_snaps_all,
            snap_s1_best=snap_s1_best,
            snap_s2_best=snap_s2_best,
            snap_stage2=snap_stage2,
        )

    # -------------------------------------------------------------------------
    # save all data to npz
    # -------------------------------------------------------------------------
    save_dict = dict(
        sweep_axis=sweep_axis,
        sweep_values=np.array(sweep_values, dtype=float),
        n_repeat=n_repeat,
        seed_base=seed_base,
        N_base=N_base,
        N_max=N_max,
        beta_base=beta_base,
        sigma2_base=sigma2_base,
        a_base=a_base,
        L_theory=L_theory,
        L_factor=L_factor,
        snap_every=snap_every,
        run_full_wasserstein=run_full_wasserstein,
        pi=pi,
        mu_list=mu_list,
    )

    for val in sweep_values:
        r   = results[val]
        key = _val_key(val)
        save_dict[f'{key}_label']              = r['label']
        save_dict[f'{key}_beta']               = r['beta']
        save_dict[f'{key}_sigma2']             = r['sigma2']
        save_dict[f'{key}_N']                  = r['N']
        save_dict[f'{key}_a']                  = r['a']
        save_dict[f'{key}_eta']                = r['eta']
        save_dict[f'{key}_L']                  = r['L']
        save_dict[f'{key}_T_run']              = r['T_run']
        save_dict[f'{key}_analytic_bayes_mse'] = r['analytic_bayes_mse']
        save_dict[f'{key}_mse_s1_opt']         = r['mse_s1_opt_runs']
        save_dict[f'{key}_mse_s2_opt']         = r['mse_s2_opt_runs']
        save_dict[f'{key}_mse_s2']             = r['mse_s2_runs']
        save_dict[f'{key}_bayes_mse']          = r['bayes_mse_runs']
        save_dict[f'{key}_mse_init']           = r['mse_init_runs']
        save_dict[f'{key}_mse_tweedie']        = r['mse_tweedie_runs']
        save_dict[f'{key}_w1_s2']              = r['w1_s2_runs']
        save_dict[f'{key}_w2_s2']              = r['w2_s2_runs']
        save_dict[f'{key}_opt_s1_depth']       = r['opt_s1_depth_runs']
        save_dict[f'{key}_opt_s2_depth']       = r['opt_s2_depth_runs']
        save_dict[f'{key}_traj_indices']       = r['traj_indices']
        save_dict[f'{key}_mse_traj_mean']      = r['mse_traj_mean']
        save_dict[f'{key}_mse_traj_std']       = r['mse_traj_std']
        save_dict[f'{key}_w1_traj_mean']       = r['w1_traj_mean']
        save_dict[f'{key}_w1_traj_std']        = r['w1_traj_std']
        save_dict[f'{key}_w2_traj_mean']       = r['w2_traj_mean']
        save_dict[f'{key}_w2_traj_std']        = r['w2_traj_std']
        save_dict[f'{key}_early_s2_traj_mean'] = r['early_s2_traj_mean']
        save_dict[f'{key}_early_s2_traj_std']  = r['early_s2_traj_std']
        save_dict[f'{key}_snap_depths']        = r['snap_depths']
        save_dict[f'{key}_z0_all']             = r['z0_all']
        save_dict[f'{key}_x_clean_all']        = r['x_clean_all']
        save_dict[f'{key}_z_snaps_all']        = r['z_snaps_all']
        save_dict[f'{key}_snap_s1_best']       = r['snap_s1_best']
        save_dict[f'{key}_snap_s2_best']       = r['snap_s2_best']
        save_dict[f'{key}_snap_stage2']        = r['snap_stage2']
        if run_full_wasserstein and r['w1_full_s2_runs'] is not None:
            save_dict[f'{key}_w1_full_s2'] = r['w1_full_s2_runs']
            save_dict[f'{key}_w2_full_s2'] = r['w2_full_s2_runs']

    t_total = time.perf_counter() - t_total_start

    npz_path = os.path.join(DIR_OUT, f'results_{run_label}.npz')
    np.savez(npz_path, **save_dict)
    print(f"\n  Saved raw data to {npz_path}")
    print(f"  Run fig4_mse_sweep_replot.py to produce figures.")
    print(f"\n  Total wall time: {t_total:.1f}s  "
          f"({len(sweep_values)} configs x {n_repeat} seeds)")
    print(f"Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a Gaussian-mixture MSE sweep.')
    parser.add_argument('--sweep', choices=['N', 'sigma2', 'beta', 'a'], default='beta',
                        help='Sweep axis (default: beta). Other settings are in main().')
    main(sweep_axis=parser.parse_args().sweep)
