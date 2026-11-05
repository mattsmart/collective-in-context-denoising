"""
Part A: 2D Gaussian mixture denoising via tied-weight attention dynamics.

Model:
    x_i ~ pi_1 * N(mu1, a1^2 * I2) + pi_2 * N(mu2, a2^2 * I2)   [clean prior]
    y_i^(0) = x_i + z_i,  z_i ~ N(0, sigma2 * I2)                [noisy input]
    y_i^(l+1) = (1-eta)*y_i^(l) + eta * sum_j a_ij^(l) * y_j^(l)
    a_ij^(l) = softmax_j(-beta/2 * ||y_i^(l) - y_j^(l)||^2)      [Gaussian RBF]

Stage 2 cross-attention:
    hat_x_i = sum_j a_ij * y_j^(L),  queries = y^(0), keys/values = y^(L)

Analytic baseline (MMSE estimator for the mixture prior):
    hat_x(y) = sum_k [ w_k(y) * (mu_k + (a_k^2/(a_k^2+sigma2)) * (y - mu_k)) ]
    where w_k(y) = pi_k * N(y; mu_k, (a_k^2+sigma2)*I2) / Z(y)

Energy landscape interpretation:
    For a frozen particle cloud Z, define
        E_beta(q; Z) = -(1/beta) * log sum_j exp(-beta/2 * ||q - z_j||^2).
    Its query gradient is q - F_beta,Z(q), where F is Gaussian attention.
    Stage 1 takes a gradient step of size eta from each current particle,
    with the cloud held fixed while evaluating that layer's update.
    As the cloud evolves, this query-dependent energy changes each layer.

    Stage 2 uses the refined cloud and beta_c = 1/sigma2:
        E2(q; Z^(L)) = -(1/beta_c) * log sum_j exp(-beta_c/2 * ||q - z_j^(L)||^2).
    The posterior mean under the particle prior is obtained by one unit
    gradient step, q - grad E2(q), starting from the original noisy query.
    The correction field hat_x(y) - y visualizes this posterior readout.

Figures produced (saved to ./fig3_output/):
    1. mse_vs_depth.png                           -- MSE vs layer, Stage 1 and Stage 2 marked
    2. energy_snapshots.png                       -- E2 energy landscape + cloud scatter across depth
    3. overview_ICL_prior_input_stage1_stage2.png -- prior / noisy input / Stage 1 / Stage 2 overview

Usage:
    python fig3_mixture_prior.py

Primary knobs: beta, a1, a2, sigma2.
eta is derived as  eta = beta * sigma2 / (2 * L_theory).
"""

import os
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ot

# ---------------------------------------------------------------------------
# global flags
# ---------------------------------------------------------------------------
# Set True to compute exact W1 (via POT Earth Mover's Distance) in addition to
# the fast sliced-W1 proxy.  Exact W1 is O(N^2 log N) and can be slow for large
# N — it is subsampled to W1_FULL_SUBSAMPLE points.  The sliced version is always
# computed regardless of this flag.
RUN_FULL_WASSERSTEIN = False
W1_FULL_SUBSAMPLE    = 500   # max particles used for exact W1 (LP solve)

# ---------------------------------------------------------------------------
# output directory
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_OUT    = os.path.join(SCRIPT_DIR, 'fig3_output')
os.makedirs(DIR_OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# global color palette
# ---------------------------------------------------------------------------
COLOR_CLOUD   = 'steelblue'    # evolving cloud z^(l)
COLOR_TARGET  = 'forestgreen'  # clean mixture samples x_i
COLOR_STAGE2  = 'crimson'      # Stage 2 cross-attention output
COLOR_BAYES   = 'darkorange'   # analytic Bayes posterior mean
TRACER_COLORS = ['#e41a1c', '#ff7f00', '#4daf4a',
                 '#984ea3', '#00bcd4', '#a65628']  # per-tracer colors (up to 6)


# ---------------------------------------------------------------------------
# numerically stable helpers
# ---------------------------------------------------------------------------

def logsumexp(z):
    z_max = np.max(z)
    return z_max + np.log(np.sum(np.exp(z - z_max)))


def softmax_rows(logits):
    """Row-wise stable softmax.  logits: (N, N) -> (N, N)."""
    logits = logits - logits.max(axis=1, keepdims=True)
    w = np.exp(logits)
    w /= w.sum(axis=1, keepdims=True)
    return w


# ---------------------------------------------------------------------------
# core dynamics  (pure numpy, 2D)
# ---------------------------------------------------------------------------

def _sq_dists(a, b):
    """
    Pairwise squared Euclidean distances between rows of a (M, d) and b (N, d).
    Returns (M, N) without materializing an (M, N, d) intermediate tensor.
    Uses ||a_i - b_j||^2 = ||a_i||^2 + ||b_j||^2 - 2 a_i·b_j, then clips to
    zero to guard against small negative values from floating-point cancellation.
    """
    a_sq = np.sum(a ** 2, axis=1, keepdims=True)   # (M, 1)
    b_sq = np.sum(b ** 2, axis=1, keepdims=True)   # (N, 1)
    dist_sq = a_sq + b_sq.T - 2.0 * (a @ b.T)     # (M, N)
    return np.maximum(dist_sq, 0.0)


def gauss_attn_nd(y, beta):
    """
    Gaussian RBF self-attention for N-dimensional particle cloud.
    y: (N, d) float
    Returns attn_out: (N, d)  where attn_out[i] = sum_j a_ij * y[j]
    a_ij = softmax_j(-beta/2 * ||y_i - y_j||^2)
    """
    dist_sq = _sq_dists(y, y)                       # (N, N), no (N,N,d) temp
    logits  = -0.5 * beta * dist_sq
    w = softmax_rows(logits)                        # (N, N)
    return w @ y                                    # (N, d)


def gauss_cross_attn_nd(query, context, beta):
    """
    Cross-attention (Stage 2): each query attends to context.
    query:   (N, d) -- original noisy inputs y^(0)
    context: (N, d) -- Stage 1 final cloud z^(L)
    Returns: (N, d)
    """
    dist_sq = _sq_dists(query, context)             # (N, N), no (N,N,d) temp
    logits  = -0.5 * beta * dist_sq
    w = softmax_rows(logits)
    return w @ context                              # (N, d)


def run_trajectory_nd(z0, beta, eta, L, beta_final=None, record_every=1):
    """
    Run the 2D attention dynamics for L layers.

    Args:
        z0:           (N, d) float -- initial noisy particles
        beta:         float -- RBF attention bandwidth
        eta:          float in (0,1) -- convex residual step size
        L:            int -- number of Stage 1 layers
        beta_final:   float -- Stage 2 cross-attn bandwidth; should be 1/sigma2
                      per the paper (beta_c = 1/sigma^2). Must be set explicitly.
        record_every: int -- record a snapshot every this many layers

    Returns:
        states:  list of (N, d) arrays  [layer 0, ..., layer L, Stage 2]
        indices: list of int            [0, ..., L, L+1]
    """
    if beta_final is None:
        raise ValueError(
            "beta_final must be set explicitly. "
            "Per the paper, use beta_final = 1.0 / sigma2 (beta_c = 1/sigma^2)."
        )

    z = z0.copy().astype(np.float64)
    states  = [z.copy()]
    indices = [0]

    for t in range(1, L + 1):
        if t % 100 == 0:
            print('Step:', t, 'of', L)
        attn_out = gauss_attn_nd(z, beta)
        z = (1.0 - eta) * z + eta * attn_out
        if (t % record_every == 0) or t == L:
            states.append(z.copy())
            indices.append(t)

    # Stage 2: cross-attention, queries = original noisy particles
    z_stage2 = gauss_cross_attn_nd(z0, z, beta_final)
    states.append(z_stage2.copy())
    indices.append(L + 1)

    return states, indices


# ---------------------------------------------------------------------------
# mixture prior utilities
# ---------------------------------------------------------------------------

def sample_mixture(pi, mu_list, a_list, N, rng):
    """
    Sample N points from a 2D isotropic Gaussian mixture.
    pi:      (K,) mixture weights (sum to 1)
    mu_list: (K, 2) component centers
    a_list:  (K,) component std devs
    Returns: x_clean (N, 2), component_labels (N,) ints
    """
    K = len(pi)
    labels = rng.choice(K, size=N, p=pi)
    x = np.zeros((N, 2))
    for k in range(K):
        mask = (labels == k)
        n_k  = int(mask.sum())
        if n_k > 0:
            x[mask] = rng.normal(0.0, a_list[k], size=(n_k, 2)) + mu_list[k]
    return x, labels


def mixture_log_density(y, pi, mu_list, a_list, sigma2=0.0):
    """
    Log-density of the mixture (optionally convolved with N(0, sigma2*I)).
    y: (N, 2) or (G, G, 2) (broadcast-safe via reshape)
    Returns: array of same shape as y[..., 0]

    If sigma2 > 0, computes density of x + N(0, sigma2*I), i.e. the noisy marginal.
    """
    shape = y.shape[:-1]
    y_flat = y.reshape(-1, 2)     # (M, 2)
    K = len(pi)
    log_comps = np.zeros((len(y_flat), K))
    for k in range(K):
        var_k = a_list[k] ** 2 + sigma2
        diff  = y_flat - mu_list[k]               # (M, 2)
        log_comps[:, k] = (np.log(pi[k])
                           - np.log(2 * np.pi * var_k)
                           - 0.5 * np.sum(diff ** 2, axis=1) / var_k)
    # log-sum-exp over components
    lse = logsumexp_rows(log_comps)               # (M,)
    return lse.reshape(shape)


def logsumexp_rows(logits):
    """Row-wise logsumexp. logits: (M, K) -> (M,)"""
    m = logits.max(axis=1, keepdims=True)
    return m.squeeze(axis=1) + np.log(np.exp(logits - m).sum(axis=1))


def mixture_posterior_mean(y, pi, mu_list, a_list, sigma2):
    """
    Exact MMSE estimator (Bayes posterior mean) for the mixture prior.
    y: (N, 2)
    Returns: (N, 2)

    hat_x(y) = sum_k [ w_k(y) * (mu_k + a_k^2/(a_k^2+sigma2) * (y - mu_k)) ]
    where w_k(y) prop to pi_k * N(y; mu_k, (a_k^2+sigma2)*I2)
    """
    N, d = y.shape
    K    = len(pi)  # num. components
    assert d == 2   # this function is specific to 2D for simplicity, but could be generalized

    log_w = np.zeros((N, K))
    for k in range(K):
        var_k = a_list[k] ** 2 + sigma2
        diff  = y - mu_list[k]                    # (N, 2)
        log_w[:, k] = (np.log(pi[k])
                       - np.log(2 * np.pi * var_k)
                       - 0.5 * np.sum(diff ** 2, axis=1) / var_k)

    log_w -= logsumexp_rows(log_w)[:, None]       # normalise in log space
    w = np.exp(log_w)                             # (N, K)

    hat_x = np.zeros((N, d))
    for k in range(K):
        shrink_k = a_list[k] ** 2 / (a_list[k] ** 2 + sigma2)
        comp_est = mu_list[k] + shrink_k * (y - mu_list[k])  # (N, 2)
        hat_x += w[:, k, None] * comp_est

    return hat_x



def stage1_energy_grid(grid_xy, cloud_z, beta, eta):
    """
    Evaluate Stage 1 energy E1(q; Z) on a grid.

    E1(q; Z) = -(1/beta) * log sum_j exp(-beta/2 * ||q - z_j||^2)

    With Z frozen, grad_q E1 = q - F_beta,Z(q). Stage 1 takes a step
    of size eta on this energy. The energy itself is independent of eta.

    grid_xy: (G*G, 2)  flattened grid points
    cloud_z: (N, 2)    current particle cloud
    Returns: (G*G,)
    """
    return stage2_energy_grid(grid_xy, cloud_z, beta)


def stage2_energy_grid(grid_xy, cloud_z, beta):
    """
    Evaluate Stage 2 energy E2(q; Z^(L)) on a grid.

    E2(q; Z^(L)) = -(1/beta) * log sum_j exp(-beta/2 * ||q - z_j||^2)

    For the Stage 2 bandwidth beta = 1/sigma2, a unit gradient step
    q - grad_q E2(q) gives the posterior mean under the particle prior.

    grid_xy: (G*G, 2)
    cloud_z: (N, 2)
    Returns: (G*G,)
    """
    n_grid = len(grid_xy)
    E = np.zeros(n_grid)
    for i in range(n_grid):
        q       = grid_xy[i]
        diff    = cloud_z - q
        dist_sq = np.sum(diff ** 2, axis=1)
        E[i]    = -(1.0 / beta) * logsumexp(-0.5 * beta * dist_sq)
    return E


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def compute_mse_nd(y_hat, x_clean):
    """MSE per particle, averaged. y_hat, x_clean: (N, d)"""
    return float(np.mean(np.sum((y_hat - x_clean) ** 2, axis=1)))


def compute_w2_nd(a, b, n_sample=1000, rng=None):
    """
    2-Wasserstein via POT (exact for empirical distributions).
    a, b: (N, d)  -- subsample to n_sample for speed if N is large.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if len(a) > n_sample:
        idx_a = rng.choice(len(a), size=n_sample, replace=False)
        idx_b = rng.choice(len(b), size=n_sample, replace=False)
        a, b  = a[idx_a], b[idx_b]
    n  = len(a)
    M  = ot.dist(a, b, metric='sqeuclidean')          # (n, n)
    w  = np.ones(n) / n
    T  = ot.emd(w, w, M)
    w2 = float(np.sqrt(np.sum(T * M)))
    return w2


def compute_w1_nd(a, b, n_projections=50, n_sample=2000, rng=None):
    """
    Sliced W1 as a proxy for 2D W1 (fast, unbiased estimator).
    a, b: (N, d)  — subsampled to n_sample each if larger.
    """
    if rng is None:
        rng = np.random.default_rng(1)
    n = min(len(a), len(b), n_sample)
    if len(a) > n:
        a = a[rng.choice(len(a), size=n, replace=False)]
    if len(b) > n:
        b = b[rng.choice(len(b), size=n, replace=False)]
    d = a.shape[1]
    total = 0.0
    for _ in range(n_projections):
        v = rng.normal(size=d)
        v /= np.linalg.norm(v)
        pa = a @ v
        pb = b @ v
        total += float(np.mean(np.abs(np.sort(pa) - np.sort(pb))))
    return total / n_projections


def compute_w1_full_nd(a, b, n_sample=None, rng=None):
    """
    Exact W1 (Earth Mover's Distance) via POT linear program.
    Uses L1 (city-block) ground metric: W1 = min_{T} sum_{ij} T_ij * ||a_i - b_j||_1.
    Subsampled to n_sample points for tractability.
    a, b: (N, d)
    """
    if rng is None:
        rng = np.random.default_rng(2)
    if n_sample is not None:
        if len(a) > n_sample:
            a = a[rng.choice(len(a), size=n_sample, replace=False)]
        if len(b) > n_sample:
            b = b[rng.choice(len(b), size=n_sample, replace=False)]
    n = len(a)
    M = ot.dist(a, b, metric='cityblock')   # L1 ground metric for W1
    w = np.ones(n) / n
    T = ot.emd(w, w, M)
    return float(np.sum(T * M))


def compute_metrics_trajectory_nd(states, x_clean, pi, mu_list, a_list,
                                  sigma2, beta, beta_cross=None, N_sample=1000, rng=None,
                                  run_full_wasserstein=None):
    """
    For each snapshot, compute MSE, W1 (sliced), W2 against a fresh clean sample.
    Also computes Bayes MSE (fixed reference).

    Two early-Stage-2 MSE curves are computed:
      'early_s2_same_beta':  cross-attn z^0 -> Z^(l) using beta (Stage 1 bandwidth).
                             Isolates cloud quality: same kernel as Stage 1, only the
                             query origin changes.
      'early_s2_cross_beta': cross-attn z^0 -> Z^(l) using beta_cross = 1/sigma2
                             (the operationally correct Stage 2 bandwidth).
                             The final Stage 2 dot is the endpoint of this curve.

    run_full_wasserstein: bool or None. If None, uses module-level RUN_FULL_WASSERSTEIN.
      When True, also computes exact W1 (via LP/POT) and exact W2; stored in
      metrics['w1_full'] and metrics['w2_full'].

    Returns:
        metrics: dict with 'mse', 'w1', 'w2', 'early_s2_same_beta',
                 'early_s2_cross_beta' lists aligned with indices
        bayes_mse: scalar -- MSE of analytic posterior mean on z0
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if beta_cross is None:
        beta_cross = 1.0 / sigma2
    if run_full_wasserstein is None:
        run_full_wasserstein = RUN_FULL_WASSERSTEIN

    # fresh clean reference sample for W1/W2
    x_ref, _ = sample_mixture(pi, mu_list, a_list, N_sample, rng)

    W2_SUBSAMPLE = 300
    mse_list = []
    w1_list  = []
    w2_list  = []
    w1_full_list = []
    w2_full_list = []
    early_s2_same_beta_list  = []
    early_s2_cross_beta_list = []
    z0 = states[0]
    for state in states:
        mse_list.append(compute_mse_nd(state, x_clean))
        w1_list.append(compute_w1_nd(state, x_ref, rng=rng))
        w2_list.append(compute_w2_nd(state, x_ref, n_sample=W2_SUBSAMPLE, rng=rng))
        if run_full_wasserstein:
            w1_full_list.append(compute_w1_full_nd(state, x_ref,
                                                   n_sample=W1_FULL_SUBSAMPLE, rng=rng))
            w2_full_list.append(compute_w2_nd(state, x_ref,
                                              n_sample=W1_FULL_SUBSAMPLE, rng=rng))
        # early Stage 2 — same beta as Stage 1 (isolates cloud quality)
        hat_same  = gauss_cross_attn_nd(z0, state, beta)
        early_s2_same_beta_list.append(compute_mse_nd(hat_same, x_clean))
        # early Stage 2 — operational beta_cross = 1/sigma2
        hat_cross = gauss_cross_attn_nd(z0, state, beta_cross)
        early_s2_cross_beta_list.append(compute_mse_nd(hat_cross, x_clean))

    # Bayes MSE: analytic posterior mean applied to z0
    hat_x_bayes = mixture_posterior_mean(z0, pi, mu_list, a_list, sigma2)
    bayes_mse   = compute_mse_nd(hat_x_bayes, x_clean)

    # Baseline: predict origin (0, 0) for all inputs
    origin_pred = np.zeros_like(x_clean)
    origin_mse  = compute_mse_nd(origin_pred, x_clean)

    # Baseline: predict the empirical mean of z0 for all inputs
    z0_mean      = z0.mean(axis=0, keepdims=True)           # (1, d)
    mean_pred    = np.broadcast_to(z0_mean, x_clean.shape)  # (N, d)
    mean_pred_mse = compute_mse_nd(mean_pred, x_clean)

    out = {'mse': mse_list, 'w1': w1_list, 'w2': w2_list,
           'early_s2_same_beta':  early_s2_same_beta_list,
           'early_s2_cross_beta': early_s2_cross_beta_list,
           # keep old key as alias so nothing else breaks
           'early_s2_mse': early_s2_same_beta_list,
           'origin_mse':   origin_mse,
           'mean_pred_mse': mean_pred_mse}
    if run_full_wasserstein:
        out['w1_full'] = w1_full_list
        out['w2_full'] = w2_full_list
    return out, bayes_mse


# ---------------------------------------------------------------------------
# plotting helpers
# ---------------------------------------------------------------------------

def _savefig(fig, fname, exts=('png',)):
    for ext in exts:
        path = os.path.join(DIR_OUT, f'{fname}.{ext}')
        fig.savefig(path, bbox_inches='tight', dpi=150)
        print(f"  Saved {path}")
    plt.close(fig)


def _mixture_contour(ax, pi, mu_list, a_list, sigma2, xlim, ylim,
                     G=120, levels=8, cmap='Greens', alpha=0.35, label=None):
    """Overlay filled contours of the clean mixture density (or noisy marginal)."""
    gx = np.linspace(xlim[0], xlim[1], G)
    gy = np.linspace(ylim[0], ylim[1], G)
    Gx, Gy = np.meshgrid(gx, gy)
    pts = np.stack([Gx, Gy], axis=-1)                 # (G, G, 2)
    Z   = np.exp(mixture_log_density(pts, pi, mu_list, a_list, sigma2=sigma2))
    cs  = ax.contourf(Gx, Gy, Z, levels=levels, cmap=cmap, alpha=alpha)
    if label:
        ax.contour(Gx, Gy, Z, levels=levels, colors='darkgreen',
                   linewidths=0.5, alpha=0.5)
    return cs


# ---------------------------------------------------------------------------
# figure 2: MSE vs depth
# ---------------------------------------------------------------------------

def plot_mse_vs_depth(states, indices, metrics, bayes_mse, label='', fname='mse_vs_depth',
                      omit_trivial_baselines=True):
    """
    MSE vs depth with four time-dependent curves + horizontal references.

    Curves (depth-dependent):
      Stage 1 cloud          -- MSE of z^(l) directly as x-prediction
      Early S2 [same beta]   -- cross-attn z^0->Z^(l) with Stage 1 beta
      Early S2 [cross beta]  -- cross-attn z^0->Z^(l) with beta_cross=1/sigma2

    Horizontal references (always shown):
      Stage 2 final          -- endpoint of the 'cross beta' curve
      Bayes optimum

    When omit_trivial_baselines=True (default):
      noisy input, predict-origin, predict-mean baselines are NOT plotted as
      lines; their values appear in the figure subtitle instead.
    When omit_trivial_baselines=False:
      those baselines are drawn as horizontal lines on the axes.
    """
    s1_mask  = [i for i, d in enumerate(indices) if d < indices[-1]]
    depths   = [indices[i] for i in s1_mask]
    mse_s1           = [metrics['mse'][i]                  for i in s1_mask]
    early_cross_beta = [metrics['early_s2_cross_beta'][i]  for i in s1_mask]
    mse_s2   = metrics['mse'][-1]
    mse_init = metrics['mse'][0]

    opt_i_s1         = int(np.argmin(mse_s1))
    opt_i_cross_beta = int(np.argmin(early_cross_beta))
    opt_dep          = depths[opt_i_s1]

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(depths, mse_s1,           '-',  color=COLOR_CLOUD,    linewidth=1.8,
            label='Stage 1 cloud')
    ax.plot(depths, early_cross_beta, '--', color='mediumpurple', linewidth=1.8,
            label='Early S2  (cross-attn z⁰→Z^(ℓ), β=1/σ²)')

    ax.axhline(mse_s2,    color='mediumpurple', linestyle=':',  linewidth=1.2,
               label=f'Stage 2 final ({mse_s2:.4f})')
    ax.axhline(bayes_mse, color=COLOR_BAYES,    linestyle=':', linewidth=1.5,
               label=f'Bayes optimum ({bayes_mse:.4f})')

    if omit_trivial_baselines:
        # Build subtitle string with trivial baseline values
        parts = [f'noisy input MSE={mse_init:.4f}']
        if 'origin_mse' in metrics:
            parts.append(f'predict-origin MSE={metrics["origin_mse"]:.4f}')
        if 'mean_pred_mse' in metrics:
            parts.append(f'predict-mean(z⁰) MSE={metrics["mean_pred_mse"]:.4f}')
        subtitle = '  |  '.join(parts)
    else:
        ax.axhline(mse_init,  color='gray', linestyle=':', linewidth=1.0,
                   label=f'noisy input ({mse_init:.4f})')
        if 'origin_mse' in metrics:
            ax.axhline(metrics['origin_mse'], color='saddlebrown', linestyle='-.', linewidth=1.0,
                       label=f'predict origin ({metrics["origin_mse"]:.4f})')
        if 'mean_pred_mse' in metrics:
            ax.axhline(metrics['mean_pred_mse'], color='teal', linestyle='-.', linewidth=1.0,
                       label=f'predict mean(z⁰) ({metrics["mean_pred_mse"]:.4f})')
        subtitle = None

    ax.axvline(opt_dep, color='black', linestyle=':', linewidth=1.0,
               label=f'$\\ell^*$ S1 = {opt_dep}')

    ax.scatter([opt_dep],                  [mse_s1[opt_i_s1]],
               s=60, color=COLOR_CLOUD,    edgecolors='black', zorder=5)
    ax.scatter([depths[opt_i_cross_beta]], [early_cross_beta[opt_i_cross_beta]],
               s=60, color='mediumpurple', edgecolors='black', zorder=5)
    # final Stage 2 dot — should sit on the cross-beta curve endpoint
    ax.scatter([depths[-1]], [mse_s2],
               s=60, color='mediumpurple', marker='D', edgecolors='black', zorder=6)

    ax.set_xlabel('Layer / depth  $\\ell$')
    ax.set_ylabel('MSE')
    title_str = f'MSE vs depth  ({label})'
    if omit_trivial_baselines and subtitle:
        title_str = f'{title_str}\n{subtitle}'
    ax.set_title(title_str, fontsize=8 if omit_trivial_baselines and subtitle else 10)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    _savefig(fig, fname)


def plot_energy_snapshots(states, indices, x_clean, pi, mu_list, a_list, sigma2,
                          beta, eta, label='', fname='energy_snapshots',
                          G=60,
                          show_e1_colormap=False,
                          flag_same_energy_clims=True,
                          n_cols=6,
                          show_tracers=True):
    """
    Two-row figure at evenly-spaced Stage 1 depths.

    Parameters
    ----------
    show_e1_colormap : bool (default False)
        Row 0: True → E1 energy colormap; False → cloud scatter + clean prior contours.
    flag_same_energy_clims : bool (default True)
        Row 1: True → shared colorbar across all columns; False → per-panel colorbars.
    n_cols : int (default 6)
        Snapshot columns at 0%, 20%, …, 100% of Stage 1 trajectory.
    show_tracers : bool (default True)
        True  → full tracer decorations (see below).  Use fname='energy_snapshots'.
        False → no tracer decorations at all.           Use fname='energy_snapshots_simple'.

    Tracer decorations (show_tracers=True)
    ---------------------------------------
    Row 0 (cloud / E1):
        Full trajectory polyline from layer 0 to current layer for each tracer.
        ● (filled circle) at layer-0 position (initial condition).
        ★ (star)          at current layer position.
        ■ (square)        at clean target x_clean.
    Row 1 (E2 absolute):
        ● at z0 (initial noisy input).
        ★ at posterior-mean update E[X|z0] evaluated with current cloud
          (i.e. what Stage 2 would output at this depth).
        ■ at clean target x_clean.
    """
    beta_cross = 1.0 / sigma2

    # ---- choose snapshot column indices ----
    s1_mask = [i for i, d in enumerate(indices) if d < indices[-1]]
    n_s1    = len(s1_mask)

    frac_positions = np.linspace(0, 1, n_cols)
    snap_si = []
    for frac in frac_positions:
        target_i = int(round(frac * (n_s1 - 1)))
        si = s1_mask[target_i]
        if si not in snap_si:
            snap_si.append(si)
    if s1_mask[0]  not in snap_si: snap_si.insert(0, s1_mask[0])
    if s1_mask[-1] not in snap_si: snap_si.append(s1_mask[-1])
    n_cols_actual = len(snap_si)

    z0 = states[0]
    norms_z0   = np.linalg.norm(z0, axis=1)
    pct_vals   = np.percentile(norms_z0, [10, 25, 42, 58, 75, 90])
    tracer_idx = [int(np.argmin(np.abs(norms_z0 - pv))) for pv in pct_vals]
    # deduplicate while preserving order (can collapse when N is very small)
    seen = set()
    tracer_idx = [t for t in tracer_idx if not (t in seen or seen.add(t))]

    all_pts = np.concatenate([z0] + [states[si] for si in snap_si], axis=0)
    pad     = 0.1
    xlim    = (all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
    ylim    = (all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)

    gx   = np.linspace(xlim[0], xlim[1], G)
    gy   = np.linspace(ylim[0], ylim[1], G)
    Gx, Gy   = np.meshgrid(gx, gy)
    grid_pts = np.stack([Gx.ravel(), Gy.ravel()], axis=1)

    # ---- pre-compute all E2 grids ----
    E2_grids = [stage2_energy_grid(grid_pts, states[si], beta=beta_cross)
                for si in snap_si]
    if flag_same_energy_clims:
        e2_vmin = min(E.min() for E in E2_grids)
        e2_vmax = max(E.max() for E in E2_grids)
    else:
        e2_vmin = e2_vmax = None

    # ---- pre-build full per-tracer trajectory arrays for Row 0 ----
    # traj_positions[k] = (n_s1, 2) array of particle k's position at every S1 layer
    if show_tracers:
        traj_positions = {
            k: np.stack([states[i][tidx] for i in s1_mask], axis=0)
            for k, tidx in enumerate(tracer_idx)
        }
        # index into s1_mask for each snap column
        snap_s1_pos = [s1_mask.index(si) for si in snap_si]

    # ---- helpers ----
    def _draw_row0_tracers(ax, col):
        """Full trajectory line up to this column + ● start + ★ current + ■ target."""
        si   = snap_si[col]
        tidx_col = snap_s1_pos[col]   # index within s1_mask (= trajectory index)
        for k, tidx in enumerate(tracer_idx):
            traj = traj_positions[k]              # (n_s1, 2)
            # polyline from layer 0 up to current depth
            ax.plot(traj[:tidx_col + 1, 0], traj[:tidx_col + 1, 1],
                    '-', color=TRACER_COLORS[k], linewidth=1.0,
                    alpha=0.55, zorder=5)
            # ● initial condition
            ax.scatter(traj[0, 0], traj[0, 1],
                       s=45, color=TRACER_COLORS[k], marker='o',
                       edgecolors='black', linewidths=0.6, zorder=7)
            # ★ current position
            ax.scatter(traj[tidx_col, 0], traj[tidx_col, 1],
                       s=60, color=TRACER_COLORS[k], marker='*',
                       edgecolors='black', linewidths=0.5, zorder=8)
            # ■ clean target
            ax.scatter(x_clean[tidx, 0], x_clean[tidx, 1],
                       s=45, color=TRACER_COLORS[k], marker='s',
                       edgecolors='black', linewidths=0.6, zorder=7)

    def _draw_energy_tracers(ax, col):
        """For Rows 1 & 2: ● z0 → ★ posterior-mean update at this depth, ■ clean target."""
        si    = snap_si[col]
        cloud = states[si]
        for k, tidx in enumerate(tracer_idx):
            # posterior mean = what Stage 2 would output at this depth for query z0[tidx]
            hat = gauss_cross_attn_nd(z0[tidx:tidx+1], cloud, beta_cross)  # (1, 2)
            # line from ● to ★
            ax.plot([z0[tidx, 0], hat[0, 0]],
                    [z0[tidx, 1], hat[0, 1]],
                    '-', color=TRACER_COLORS[k], linewidth=1.0,
                    alpha=0.6, zorder=6)
            # ● initial noisy input
            ax.scatter(z0[tidx, 0], z0[tidx, 1],
                       s=45, color=TRACER_COLORS[k], marker='o',
                       edgecolors='black', linewidths=0.6, zorder=7)
            # ★ posterior mean update
            ax.scatter(hat[0, 0], hat[0, 1],
                       s=60, color=TRACER_COLORS[k], marker='*',
                       edgecolors='black', linewidths=0.5, zorder=8)
            # ■ clean target
            ax.scatter(x_clean[tidx, 0], x_clean[tidx, 1],
                       s=45, color=TRACER_COLORS[k], marker='s',
                       edgecolors='black', linewidths=0.6, zorder=7)

    # ---- draw ----
    panel_w = 2.8
    fig, axes = plt.subplots(2, n_cols_actual,
                             figsize=(panel_w * n_cols_actual, panel_w * 2 + 1),
                             squeeze=False)

    for col, si in enumerate(snap_si):
        cloud = states[si]
        depth = indices[si]

        # ---- Row 0: cloud or E1 ----
        ax = axes[0][col]
        if show_e1_colormap:
            E1  = stage1_energy_grid(grid_pts, cloud, beta=beta, eta=eta)
            cs1 = ax.contourf(Gx, Gy, E1.reshape(G, G),
                              levels=30, cmap='Spectral_r', alpha=0.85)
            plt.colorbar(cs1, ax=ax, shrink=0.8, pad=0.02)
            row0_title = (
                f'S1  $\\ell={depth}$  (β={beta})\n'
                r'$E_1 = -\frac{1}{\beta}\log\Sigma e^{-\beta/2\|q-z_j\|^2}$'
            )
        else:
            _mixture_contour(ax, pi, mu_list, a_list, sigma2=0.0,
                             xlim=xlim, ylim=ylim, levels=6,
                             cmap='Greens', alpha=0.20)
            ax.scatter(cloud[:, 0], cloud[:, 1],
                       s=2, color=COLOR_CLOUD, alpha=0.25,
                       linewidths=0, rasterized=True)
            row0_title = f'Cloud  $\\ell={depth}$'

        if show_tracers:
            _draw_row0_tracers(ax, col)

        ax.set_title(row0_title, fontsize=7)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])

        # ---- Row 1: E2 absolute ----
        ax  = axes[1][col]
        E2  = E2_grids[col]
        kwargs_cf = dict(levels=30, cmap='Spectral_r', alpha=0.85)
        if flag_same_energy_clims:
            kwargs_cf['vmin'] = e2_vmin
            kwargs_cf['vmax'] = e2_vmax
        cs2 = ax.contourf(Gx, Gy, E2.reshape(G, G), **kwargs_cf)
        if not flag_same_energy_clims:
            plt.colorbar(cs2, ax=ax, shrink=0.8, pad=0.02)

        ax.scatter(cloud[:, 0], cloud[:, 1],
                   s=2, color='white', alpha=0.15, linewidths=0, rasterized=True)

        if show_tracers:
            _draw_energy_tracers(ax, col)

        ax.set_title(f'$E_2$  $\\ell={depth}$  (β_c=1/σ²={beta_cross:.2f})', fontsize=7)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])

    # ---- shared colorbars ----
    if flag_same_energy_clims:
        sm2 = plt.cm.ScalarMappable(
            cmap='Spectral_r',
            norm=plt.Normalize(vmin=e2_vmin, vmax=e2_vmax))
        sm2.set_array([])
        fig.colorbar(sm2, ax=axes[1, :].tolist(),
                     shrink=0.6, pad=0.02, label='$E_2$')

    row0_desc = ('E1 colormap' if show_e1_colormap
                 else 'cloud KDE  (green = clean prior)')
    tracer_desc = (
        'Row 0 tracers: ● z0  ★ current  ■ target  (line = full trajectory).  '
        'Row 1 tracers: ● z0  ★ E[X|z0] at this depth  ■ target.'
        if show_tracers else 'No tracer decorations.'
    )
    caption = (
        f'Row 0: {row0_desc}.  '
        f'Row 1: E2 absolute (β_c=1/σ²={beta_cross:.2f}).  '
        + tracer_desc
    )
    fig.suptitle(f'{label}\n{caption}', fontsize=8, y=1.01)
    fig.tight_layout()
    #_savefig(fig, fname, exts=('png', 'svg'))
    _savefig(fig, fname, exts=('png',))


# ---------------------------------------------------------------------------
# figure 8: overview — prior / Stage 1 cloud / Stage 2 denoising
# ---------------------------------------------------------------------------

def plot_overview_icl(states, indices, x_clean, z0, pi, mu_list, a_list,
                      sigma2, beta, metrics, label='',
                      fname='overview_prior_input_stage1_stage2',
                      n_s1_cols=4, n_annotated=5, seed_annot=7,
                      show_tracers=True):
    """
    2 × (1 + n_s1_cols) overview figure.

    Layout
    ------
    Row 0 — clean prior + Stage 1 cloud evolution:
        col 0              : clean prior x_clean  (deepskyblue scatter)
        cols 1 .. n_s1_cols: Stage 1 cloud z^(ℓ) at evenly-spaced depths
                             (steelblue scatter)

    Row 1 — noisy input + Stage 2 denoising at matching depths:
        col 0              : noisy input z^(0)  (#da8319 scatter)
                             n_annotated points marked as ○ circles,
                             each connected by a thin dashed black line
                             to its clean target ■ (square).
                             σ² annotated in corner.
        cols 1 .. n_s1_cols: Stage 2 cross-attention output using the S1
                             cloud from the matching column as context.
                             Purple (#7b2d8b) scatter; same annotated
                             particles shown as ★ stars.

    Every panel is annotated with MSE in the top-right corner.
    Annotated particles are consistent across all panels.
    """
    COLOR_PRIOR = 'deepskyblue'
    COLOR_NOISY = '#da8319'
    COLOR_S1    = 'steelblue'
    COLOR_S2    = '#7b2d8b'

    S_SQUARE = 28   # marker size for ■ targets (~50% of old 55)
    S_CIRCLE = 38   # marker size for ○ initial positions
    S_STAR   = 60   # marker size for ★ current/predicted positions

    beta_final = 1.0 / sigma2

    # ---- choose n_s1_cols evenly-spaced S1 snapshot state-list indices ----
    s1_mask = [i for i, d in enumerate(indices) if d < indices[-1]]
    n_s1    = len(s1_mask)
    frac_pos = np.linspace(0, 1, n_s1_cols)
    snap_si  = []
    for frac in frac_pos:
        target_i = int(round(frac * (n_s1 - 1)))
        si = s1_mask[target_i]
        if si not in snap_si:
            snap_si.append(si)
    if s1_mask[-1] not in snap_si:
        snap_si[-1] = s1_mask[-1]
    snap_si = snap_si[:n_s1_cols]

    snap_depths  = [indices[si] for si in snap_si]
    snap_s1_pos  = [s1_mask.index(si) for si in snap_si]   # position in s1_mask

    # ---- pick annotated particles and assign per-particle colors ----
    rng_a = np.random.default_rng(seed_annot)
    n_ann = min(n_annotated, len(z0))
    annot_idx   = rng_a.choice(len(z0), size=n_ann, replace=False)
    annot_colors = [TRACER_COLORS[k % len(TRACER_COLORS)] for k in range(n_ann)]

    # ---- full S1 trajectories for each annotated particle ----
    # traj[k] shape (n_s1, 2): position of annot_idx[k] at every recorded S1 layer
    traj = {k: np.stack([states[i][annot_idx[k]] for i in s1_mask], axis=0)
            for k in range(n_ann)}

    # ---- axis limits ----
    all_pts = np.concatenate([x_clean, z0], axis=0)
    pad  = 0.5
    xlim = (all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
    ylim = (all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)

    # ---- pre-compute Stage 2 outputs at each S1 depth ----
    s2_clouds = [gauss_cross_attn_nd(z0, states[si], beta_final) for si in snap_si]

    mse_noisy = compute_mse_nd(z0, x_clean)
    mse_s1    = [compute_mse_nd(states[si], x_clean) for si in snap_si]
    mse_s2    = [compute_mse_nd(s2c, x_clean) for s2c in s2_clouds]

    # ---- shared helpers ----
    n_cols = 1 + n_s1_cols

    def _panel_base(ax, pts, color, alpha=0.18, s=3):
        ax.scatter(pts[:, 0], pts[:, 1],
                   s=s, color=color, alpha=alpha,
                   linewidths=0, rasterized=True)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ['top', 'right', 'left', 'bottom']:
            ax.spines[sp].set_visible(False)

    def _annotate_mse(ax, mse_val):
        ax.text(0.97, 0.97, f'MSE={mse_val:.4f}',
                transform=ax.transAxes, fontsize=6.5,
                va='top', ha='right', family='monospace',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.75, lw=0))

    def _draw_targets(ax):
        """■ clean targets, colored per particle."""
        for k, ai in enumerate(annot_idx):
            ax.scatter(x_clean[ai, 0], x_clean[ai, 1],
                       s=S_SQUARE, color=annot_colors[k], marker='s',
                       edgecolors='black', linewidths=0.6, zorder=8)

    # ---- figure ----
    panel_w = 2.6
    fig, axes = plt.subplots(2, n_cols,
                             figsize=(panel_w * n_cols, panel_w * 2 + 0.6),
                             squeeze=False)

    # ===== Row 0 =====

    # col 0: clean prior — ■ squares at clean positions, per-particle color
    ax = axes[0][0]
    _panel_base(ax, x_clean, COLOR_PRIOR, alpha=0.22, s=4)
    if show_tracers:
        _draw_targets(ax)
    _annotate_mse(ax, 0.0)
    ax.set_title(r'Clean prior  $x \sim p(x)$', fontsize=8)

    # cols 1..: Stage 1 clouds — full trajectory polyline + ○ start + ★ current + ■ target
    for c, (si, depth, mse, s1pos) in enumerate(
            zip(snap_si, snap_depths, mse_s1, snap_s1_pos)):
        ax = axes[0][c + 1]
        cloud = states[si]
        _panel_base(ax, cloud, COLOR_S1, alpha=0.20, s=3)
        if show_tracers:
            for k, ai in enumerate(annot_idx):
                col = annot_colors[k]
                # full trajectory from layer 0 up to this column's depth
                ax.plot(traj[k][:s1pos + 1, 0], traj[k][:s1pos + 1, 1],
                        '-', color=col, linewidth=1.0, alpha=0.55, zorder=5)
                # ○ initial position (z0)
                ax.scatter(traj[k][0, 0], traj[k][0, 1],
                           s=S_CIRCLE, color=col, marker='o',
                           edgecolors='black', linewidths=0.6, zorder=7)
                # ★ current position
                ax.scatter(traj[k][s1pos, 0], traj[k][s1pos, 1],
                           s=S_STAR, color=col, marker='*',
                           edgecolors='black', linewidths=0.5, zorder=8)
            _draw_targets(ax)
        _annotate_mse(ax, mse)
        ax.set_title(f'Stage 1  $\\ell={depth}$', fontsize=8)

    # ===== Row 1 =====

    # col 0: noisy input — ○ circles per-particle, dashed line to ■ target
    ax = axes[1][0]
    _panel_base(ax, z0, COLOR_NOISY, alpha=0.20, s=3)
    if show_tracers:
        for k, ai in enumerate(annot_idx):
            col = annot_colors[k]
            ax.plot([z0[ai, 0], x_clean[ai, 0]],
                    [z0[ai, 1], x_clean[ai, 1]],
                    '--', color=col, linewidth=0.8, alpha=0.6, zorder=6)
            ax.scatter(z0[ai, 0], z0[ai, 1],
                       s=S_CIRCLE, color=col, marker='o',
                       edgecolors='black', linewidths=0.6, zorder=8)
        _draw_targets(ax)
    ax.text(0.03, 0.97, f'$\\sigma^2={sigma2}$',
            transform=ax.transAxes, fontsize=7, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.75, lw=0))
    _annotate_mse(ax, mse_noisy)
    ax.set_title(r'Noisy input  $z^{(0)} = x + \epsilon$', fontsize=8)

    # cols 1..: Stage 2 outputs — ○ at z0, ★ at prediction, ■ target
    for c, (s2c, depth, mse) in enumerate(zip(s2_clouds, snap_depths, mse_s2)):
        ax = axes[1][c + 1]
        _panel_base(ax, s2c, COLOR_S2, alpha=0.20, s=3)
        if show_tracers:
            for k, ai in enumerate(annot_idx):
                col = annot_colors[k]
                # line from ○ to ★
                ax.plot([z0[ai, 0], s2c[ai, 0]],
                        [z0[ai, 1], s2c[ai, 1]],
                        '-', color=col, linewidth=1.0, alpha=0.55, zorder=5)
                # ○ original noisy position
                ax.scatter(z0[ai, 0], z0[ai, 1],
                           s=S_CIRCLE, color=col, marker='o',
                           edgecolors='black', linewidths=0.6, zorder=7)
                # ★ Stage 2 prediction
                ax.scatter(s2c[ai, 0], s2c[ai, 1],
                           s=S_STAR, color=col, marker='*',
                           edgecolors='black', linewidths=0.5, zorder=8)
            _draw_targets(ax)
        _annotate_mse(ax, mse)
        ax.set_title(f'Stage 2  (S1 $\\ell={depth}$)', fontsize=8)

    if show_tracers:
        fig.suptitle(
            f'ICL denoising overview  |  {label}\n'
            r'Row 0: prior $p(x)$ then Stage 1 cloud $z^{(\ell)}$'
            r'  (line = particle trajectory, ○ = $z^{(0)}$, ★ = current, ■ = target).  '
            r'Row 1: noisy input $z^{(0)}$ then Stage 2 output'
            r'  (○ = $z^{(0)}$, ★ = prediction, ■ = target).',
            fontsize=7.5)
    else:
        fig.suptitle(
            f'ICL denoising overview  |  {label}\n'
            r'Row 0: prior $p(x)$ then Stage 1 cloud $z^{(\ell)}$.  '
            r'Row 1: noisy input $z^{(0)}$ then Stage 2 output.',
            fontsize=7.5)
    fig.tight_layout()
    _savefig(fig, fname, exts=('png',))


# ---------------------------------------------------------------------------
# single-run experiment runner
# ---------------------------------------------------------------------------

def run_experiment(N, pi, mu_list, a_list, sigma2, beta, eta, L,
                   beta_final=None, seed=42, record_every=1, verbose=True,
                   run_full_wasserstein=None):
    """
    Single run of the 2D mixture experiment.
    run_full_wasserstein: if None, uses module-level RUN_FULL_WASSERSTEIN flag.
    Returns (states, indices, metrics, bayes_mse, x_clean, z0, labels_clean).
    """
    rng = np.random.default_rng(seed)

    x_clean, labels_clean = sample_mixture(pi, mu_list, a_list, N, rng)
    noise = rng.normal(0.0, np.sqrt(sigma2), size=(N, 2))
    z0    = x_clean + noise

    if beta_final is None:
        beta_final = 1.0 / sigma2

    states, indices = run_trajectory_nd(
        z0, beta=beta, eta=eta, L=L,
        beta_final=beta_final,
        record_every=record_every,
    )

    rng_m = np.random.default_rng(seed + 1)
    metrics, bayes_mse = compute_metrics_trajectory_nd(
        states, x_clean, pi, mu_list, a_list, sigma2, beta,
        beta_cross=beta_final,
        N_sample=min(N, 2000), rng=rng_m,
        run_full_wasserstein=run_full_wasserstein,
    )

    if verbose:
        mse_init = metrics['mse'][0]
        s1_mses  = [metrics['mse'][i] for i, d in enumerate(indices) if d < indices[-1]]
        opt_dep  = indices[int(np.argmin(s1_mses))]
        mse_s2   = metrics['mse'][-1]
        print(f"  beta={beta}  eta={eta:.5f}  L={L}  sigma2={sigma2}")
        print(f"    MSE: init={mse_init:.4f}  opt-S1={min(s1_mses):.4f}  "
              f"Stage2={mse_s2:.4f}  Bayes={bayes_mse:.4f}  (opt_l={opt_dep})")

    return states, indices, metrics, bayes_mse, x_clean, z0, labels_clean


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    # =========================================================================
    # Select active configuration
    # =========================================================================
    ACTIVE_CONFIG = 'circle'   # 'explicit' | 'circle'

    # -------------------------------------------------------------------------
    # (1) explicit — 2-component symmetric mixture (used for Fig 4)
    # -------------------------------------------------------------------------
    if ACTIVE_CONFIG == 'explicit':
        mu_list = np.array([[-1.0, 0.0], [1.0, 0.0]])
        a_list  = np.array([0.1,  0.1])    # per-component std dev
        pi      = np.array([0.5,  0.5])    # mixture weights

    # -------------------------------------------------------------------------
    # (2) circle — K components on a ring of given radius (used for Fig 3)
    # -------------------------------------------------------------------------
    elif ACTIVE_CONFIG == 'circle':
        K          = 6
        radius     = 1.0
        a_comp     = 0.05       # std dev shared by all components
        equal_weights = False   # False → alternating strong/weak weights
        random_angles = False   # True → random angles (seed below); False → evenly spaced
        angle_seed = 123

        angles = (np.random.default_rng(angle_seed).uniform(0, 2 * np.pi, K)
                  if random_angles
                  else np.linspace(0, 2 * np.pi, K, endpoint=False))
        mu_list = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
        a_list  = np.full(K, a_comp)
        if equal_weights:
            pi = np.full(K, 1.0 / K)
        else:
            # alternating strong / weak
            raw = np.where(np.arange(K) % 2 == 0, 0.10, 0.02)
            pi  = raw / raw.sum()

    else:
        raise ValueError(f"Unknown ACTIVE_CONFIG: {ACTIVE_CONFIG!r}")

    pi = np.asarray(pi, dtype=float)  # ensure float and normalize component weights
    pi /= pi.sum()

    # =========================================================================
    # Dynamics
    # =========================================================================
    sigma2     = 0.1     # isotropic noise variance
    N          = 5000
    beta       = 20.0
    L_max      = 60      # we use low layer count for the figure; theory wants small steps (many layers)
    L_theory   = L_max // 3  # defined by first-order theory
    L          = L_max
    eta        = beta * sigma2 / (2 * L_theory)
    beta_final = None    # Stage 2 beta; None => 1/sigma2 (theory choice)
    seed       = 0
    # =========================================================================

    h        = eta / beta
    T_theory = L_theory * h
    T_run    = L * h

    K_mix = len(pi)
    a_summary = f'a={a_list[0]}' if len(set(a_list)) == 1 else f'a=[{a_list[0]:.2f}..{a_list[-1]:.2f}]'
    label = (f'N={N}  K={K_mix}  β={beta}  {a_summary}  '
             f'σ²={sigma2}  L_theory={L_theory}  L={L}')

    print(f"\n=== Fig. 3: 2D Gaussian Mixture ===")
    print(f"  {label}")
    print(f"  eta={eta:.5f}  h={h:.6f}  T_theory={T_theory:.4f}  T_run={T_run:.4f}")

    print("\n[Running single experiment...]")
    t0 = time.time()
    states, indices, metrics, bayes_mse, x_clean, z0, labels_clean = run_experiment(
        N=N, pi=pi, mu_list=mu_list, a_list=a_list,
        sigma2=sigma2, beta=beta, eta=eta, L=L,
        beta_final=beta_final, seed=seed,
        record_every=1, verbose=True,
    )
    print(f"  dynamics done  {time.time() - t0:.1f}s")

    stage2_out = states[-1]

    print("[Figure: ICL overview — prior / Stage 1 / Stage 2]")
    plot_overview_icl(
        states, indices, x_clean, z0, pi, mu_list, a_list,
        sigma2=sigma2, beta=beta, metrics=metrics, label=label,
        fname='overview_prior_input_stage1_stage2',
        show_tracers=False,
    )
    plot_overview_icl(
        states, indices, x_clean, z0, pi, mu_list, a_list,
        sigma2=sigma2, beta=beta, metrics=metrics, label=label,
        fname='overview_prior_input_stage1_stage2_tracers',
        show_tracers=True,
    )

    print("[Figure: MSE vs depth]")
    plot_mse_vs_depth(
        states, indices, metrics, bayes_mse,
        label=label, fname='mse_vs_depth',
    )

    print("[Figure: energy snapshots — simple (no tracers)]")
    plot_energy_snapshots(
        states, indices, x_clean, pi, mu_list, a_list, sigma2,
        beta=beta, eta=eta, label=label, fname='energy_snapshots_particle_prior',
        G=60, show_tracers=False,
    )

    print("[Figure: energy snapshots — with tracer decorations]")
    plot_energy_snapshots(
        states, indices, x_clean, pi, mu_list, a_list, sigma2,
        beta=beta, eta=eta, label=label, fname='energy_snapshots_tracers',
        G=60, show_tracers=True,
    )

    # -------------------------------------------------------------------------
    # Save npz for replotting
    # -------------------------------------------------------------------------
    # Stage 1 metric trajectories (exclude the Stage 2 final entry)
    s1_mask      = [i for i, d in enumerate(indices) if d < indices[-1]]
    traj_indices = np.array([indices[i] for i in s1_mask])
    mse_traj     = np.array([metrics['mse'][i]                 for i in s1_mask])
    w1_traj      = np.array([metrics['w1'][i]                  for i in s1_mask])
    w2_traj      = np.array([metrics['w2'][i]                  for i in s1_mask])
    early_s2_same_traj  = np.array([metrics['early_s2_same_beta'][i]  for i in s1_mask])
    early_s2_cross_traj = np.array([metrics['early_s2_cross_beta'][i] for i in s1_mask])

    npz_path = os.path.join(DIR_OUT, 'run_data.npz')
    np.savez(
        npz_path,
        # experiment params
        label=label,
        pi=pi,
        mu_list=mu_list,
        a_list=a_list,
        sigma2=sigma2,
        beta=beta,
        eta=eta,
        L=L,
        L_theory=L_theory,
        T_theory=T_theory,
        T_run=T_run,
        seed=seed,
        N=N,
        # particles
        x_clean=x_clean,
        z0=z0,
        labels_clean=labels_clean,
        stage2_out=stage2_out,
        # metric trajectories (Stage 1 only)
        traj_indices=traj_indices,
        mse_traj=mse_traj,
        w1_traj=w1_traj,
        w2_traj=w2_traj,
        early_s2_same_traj=early_s2_same_traj,
        early_s2_cross_traj=early_s2_cross_traj,
        mse_stage2=np.float64(metrics['mse'][-1]),
        w1_stage2=np.float64(metrics['w1'][-1]),
        w2_stage2=np.float64(metrics['w2'][-1]),
        bayes_mse=np.float64(bayes_mse),
        origin_mse=np.float64(metrics.get('origin_mse', float('nan'))),
        mean_pred_mse=np.float64(metrics.get('mean_pred_mse', float('nan'))),
    )
    print(f"\n  Saved run data to {npz_path}")
    print(f"\nDone. Outputs in: {DIR_OUT}")


if __name__ == '__main__':
    main()
