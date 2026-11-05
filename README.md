# collective-in-context-denoising

[![arXiv](https://img.shields.io/badge/arXiv-2605.29351-b31b1b.svg)](https://arxiv.org/abs/2605.29351)

Code for **Attention as In-Context Empirical Bayes: A Two-Stage View via Particle Dynamics** (NeurIPS 2026).

This repository reproduces the synthetic experiments in Figures 2-5. Stage 1 refines a particle prior with repeated Gaussian self-attention; Stage 2 uses the original noisy inputs as queries for posterior averaging. The experiments use forward simulations, with no model training or external datasets.

#### Dependencies

Tested with **Python 3.12**. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The requirements pin the tested versions of NumPy, Matplotlib, POT (Python Optimal Transport), and SciPy, which POT requires.

#### Figure generation

Run all commands from the repository root. Scripts generate their data and save plots in the output folders listed below.

**Figure 2: Gaussian variance decay**

```bash
python scripts/fig2_variance_decay_beta_run.py
python scripts/fig2_variance_decay_beta_replot.py
```

Output: `scripts/fig2_output_variance_decay/`. The variance-decay plots provide the numerical panel of Figure 2.

**Figure 3: Gaussian-mixture example**

```bash
python scripts/fig3_mixture_prior.py
```

Output: `scripts/fig3_output/`. The `energy_snapshots_particle_prior.png` plot shows the particle and energy landscapes; the script also generates overview and MSE plots.

**Figures 4 and 5: MSE sweeps**

Run the simulation and then replot its saved data for each sweep:

```bash
# Context length: Figures 4(a) and 5(a)
python scripts/fig4_mse_sweep_run.py --sweep N
python scripts/fig4_mse_sweep_replot.py --sweep N

# Noise variance: Figure 4(b)
python scripts/fig4_mse_sweep_run.py --sweep sigma2
python scripts/fig4_mse_sweep_replot.py --sweep sigma2

# Stage 1 bandwidth: Figure 5(b)
python scripts/fig4_mse_sweep_run.py --sweep beta
python scripts/fig4_mse_sweep_replot.py --sweep beta
```

Output: `scripts/fig4_sweep_output/`. The main plots are `mse_vs_N.png`, `mse_vs_sigma2.png`, `mse_s1s2_combined_N.png`, and `mse_vs_beta.png`. Additional diagnostic plots are generated alongside them.

The **Stage 2 only (no prior refinement)** baseline applies Stage 2 directly to the noisy context at depth zero. It uses the observation-noise scale and is distinct from a shrinking-bandwidth KDE-Tweedie estimator. Existing data files retain the legacy key `mse_tweedie` for this baseline.

#### Settings and outputs

Experiment settings, including context size, depth, noise variance, and random seeds, are in `main()` in each simulation script. The default Figure 2 experiment uses 20 seeds; the Figure 4-5 sweeps use 8. For a smaller trial run, lower `N` (or `N_base`) and `n_repeat`; for the context-length sweep, also shorten its list of `N` values.

The calculations run on CPU and use dense attention matrices, so runtime and memory increase with context size. Simulation scripts save `.npz` data so plots can be regenerated without repeating the simulations. Repeating a command overwrites that experiment's generated files. The paper combines and formats selected plots into its final figure panels.

#### Scripts

- `fig2_variance_decay_beta_run.py`: Gaussian variance-decay simulations.
- `fig2_variance_decay_beta_replot.py`: Replot the Gaussian simulations.
- `fig3_mixture_prior.py`: Gaussian-mixture dynamics, posterior readout, baselines, and Figure 3 plots. Also provides the shared functions used by the sweep scripts.
- `fig4_mse_sweep_run.py`: Context-length, noise, and bandwidth sweeps.
- `fig4_mse_sweep_replot.py`: MSE summaries, depth curves, and diagnostic plots.

#### Related work

Code for the earlier ICML 2025 paper is in [in-context-denoising](https://github.com/mattsmart/in-context-denoising).
