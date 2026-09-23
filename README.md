<div align="center">

# S2MAM: Semi-supervised Meta Additive Models

**Bilevel Feature Selection with Exact Implicit Hypergradients for Partially-Labeled Data**

[![English](https://img.shields.io/badge/Language-English-blue?style=for-the-badge)](README.md) [![简体中文](https://img.shields.io/badge/Language-简体中文-red?style=for-the-badge)](README_zh-CN.md)

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Tests-29%20passed-brightgreen?logo=pytest&logoColor=white)](tests/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

</div>

---

S2MAM is the official PyTorch implementation of **Semi-supervised Meta Additive Models (S2MAM)**, a bilevel-optimization framework for feature selection with **partially labeled** samples, **massive redundant dimensions** and **extremely noisy features**:

- **Lower level**: a semi-supervised additive model — B-spline basis per feature + smoothness penalty + **graph Laplacian regularization** over labeled & unlabeled samples, solved in **closed form**.
- **Upper level**: feature-selection scores `m ∈ [0,1]^p` under a cardinality constraint, optimized by **first-order projected gradient** with an **exact implicit hypergradient** (autograd through the closed-form lower solution).

> **Difference from PBCS.** In [Probabilistic Bilevel Coreset Selection (PBCS)](https://github.com/x-zho14/Probabilistic-Bilevel-Coreset-Selection) the lower model is a SGD-trained neural network, so the hypergradient must be approximated by a one-step pseudo-update + Hessian-vector product (biased). In S2MAM the lower level (partial labels + Laplacian-regularized additive model) is a strongly convex quadratic problem with a closed-form solution, which yields an **exact hypergradient** and convergence of the upper-level projected-gradient iteration on the convex compact set `{m ∈ [0,1]^p, Σm = k}`. PBCS-style REINFORCE machinery is retained as an alternative engine for lower models without differentiable closed forms (e.g., LapSVM on non-additive data such as two-moons).

## 📑 Table of Contents

- [✨ Highlights](#-highlights)
- [📢 News](#-news)
- [🔧 Installation](#-installation)
- [⚡ Quick Start](#-quick-start)
- [🏗️ Architecture](#️-architecture)
- [⚙️ Configuration](#️-configuration)
- [📁 Experimental Results](#-experimental-results)
- [✅ Tests](#-tests)
- [☑️ Todo](#️-todo)
- [🤝 Acknowledgements](#-acknowledgements)
- [📖 Citation](#-citation)
- [📄 License](#-license)

## ✨ Highlights

- **Exact implicit hypergradient** — the lower level is solved in closed form (`torch.linalg.solve`), and autograd differentiates *through* the solution, giving the true hypergradient `∂F/∂m`.
- **Analytical LOO-CV upper objective** — leave-one-out residuals of the linear smoother computed via hat-matrix diagonals `h_ii = (Φ_l * (Φ_l A⁻¹)).sum(1)/l`; no validation split, no test-set leakage.
- **Gating penalty** `δ(1−m_j)²‖β_j‖²` — breaks the attenuation–reoptimization invariance (envelope flatness) of masked additive models, ensuring the soft solution is consistent with the discrete top-k selection.
- **PBCS-style engineering, adapted** — REINFORCE Bernoulli estimator with control-variate variance reduction, gradient clipping, cosine learning-rate decay, bisection cardinality projection, and ts/te cubic-eased *iterative pruning* schedule (start from all features, prune gradually).
- **Greedy swap refinement** — after top-k discretization, pairwise `selected ↔ unselected` swaps are evaluated on the exact upper objective to recover weak-signal features.
- **Dual engines** — `implicit` (default; exact hypergradient) and `reinforce` (PBCS-style value feedback, automatically selected when the lower model is LapSVM, e.g., non-additive moon data).
- **Robustness by design** — validated under partial labeling (25% labeled), 92–98% redundant dimensions and extreme noisy features `N(100,100)`.

## 📢 News

- **[2026-09]** Code refactored: differentiable lower level (`model/additive.py`), first-order projected-gradient upper level (`model/bilevel.py`), end-to-end pipeline (`model/experiment.py`), full unit + functional test suite (`tests/`, 29 tests passing).
- **[2026-09]** Fixed original-repo bugs: module-level `argparse` side effects, undefined variables in demo drivers, Laplacian construction bug in `LapRLS`, closed-form models being refit in a 1000-iteration loop.

## 🔧 Installation

```bash
# 1. Clone the repository
git clone https://github.com/zxlml/S2MAM.git
cd S2MAM

# 2. Create environment
conda create -n s2mam python=3.10 -y
conda activate s2mam

# 3. Install dependencies
pip install torch numpy scipy scikit-learn pandas openpyxl pytest
```

> **Note** — everything also runs on CPU-only machines (the default closed-form lower solver is float64 CPU); a GPU is optional.
>
> If you hit `OMP: Error #15 (libiomp5md.dll already initialized)` on Windows, set `KMP_DUPLICATE_LIB_OK=TRUE`.

## ⚡ Quick Start

Run the four synthetic studies of the paper (partial labeling + noisy dimensions) from the repo root:

```bash
# (2) Synthetic additive regression: 8 informative + 92 redundant + 10 noisy dims, 50/200 labeled
python main.py --task regression --dataset additive

# (1) Friedman regression: 5 informative + 95 redundant + 10 noisy dims
python main.py --task regression --dataset friedman

# (3) Synthetic additive classification: 2 informative + 98 redundant + 10 noisy dims, 10 labels/class
python main.py --task classification --dataset additive

# (4) Two-moons classification (non-additive -> LapSVM lower + REINFORCE engine)
python main.py --task classification --dataset moon --engine reinforce --max_outer_iter 40
```

Each run prints a report with the selected features, test metric (MSE / accuracy), and the **full-feature** and **oracle** baselines. Programmatic usage:

```python
from model.experiment import ExperimentConfig, run_experiment, report

cfg = ExperimentConfig(task='regression', dataset='additive',
                       N=200, Ntest=200, n_labeled=50, n_noisy=10,
                       noise_std=1.0, seed=0)
out = run_experiment(cfg)
report(out)
```

### Run the tests

```bash
python -m pytest tests -q        # 29 unit + functional tests
```

## 🏗️ Architecture

```
S2MAM/
├── main.py                    # CLI entry (regression / classification demos)
├── model/
│   ├── additive.py            # Lower level: differentiable closed-form semi-supervised
│   │                          #   additive model (spline expansion, smoothness penalty,
│   │                          #   graph Laplacian, gating penalty, analytical LOO)
│   ├── bilevel.py             # Upper level: first-order projected gradient
│   │                          #   (exact implicit hypergradient / REINFORCE dual engine,
│   │                          #    bisection projection, cosine LR, iterative pruning)
│   ├── experiment.py          # End-to-end pipeline + baselines + greedy swap refinement
│   ├── models.py              # demo_r / demo_c / CLI parsing (side-effect free)
│   ├── progress_r.py          # LapRLS / LapSVM / RBF kernel (original repo, bugs fixed)
│   └── progress_c.py          # Classification-side compatibility layer
├── data/
│   ├── data_generation.py     # Paper simulation data: partial labels + N(100,100) noise dims
│   └── data.md                # Dataset descriptions from the paper
├── tests/                     # 29 unit + functional tests (pytest)
│   ├── test_additive.py       # Closed-form correctness, finite-difference hypergradient
│   ├── test_bilevel.py        # Projection, schedule, both engines
│   ├── test_data.py           # Paper-faithful simulation data
│   └── test_functional.py     # End-to-end: selection quality vs. oracle / full baselines
├── hypergrad/                 # Hypergradient utilities (from original repo)
├── coreset_utils/             # Utility functions (from original repo)
├── reinforce_utils/           # REINFORCE helpers (from original repo)
└── logging_utils/             # Logging / directory management (from original repo)
```

## ⚙️ Configuration

All behavior is controlled by `ExperimentConfig` (or equivalently the CLI). Key options:

| Option | Default | Description |
| --- | --- | --- |
| `task` | `'regression'` | `regression` or `classification` |
| `dataset` | `'additive'` | Regression: `additive`, `friedman`; Classification: `additive`, `moon` |
| `n_labeled` / `labeled_per_class` | `50` / `10` | Partial-labeling budget (regression / classification) |
| `n_noisy` | `10` | Number of extreme noisy features `N(100,100)` appended |
| `upper_obj` | `'loo'` | Upper objective: `loo` = analytical leave-one-out (no leakage), `test` = original-repo protocol |
| `gate_delta` | `1.0` | Gating penalty coefficient `δ(1−m_j)²` |
| `r` | `4` / `5` | B-spline basis functions per feature (regression / classification) |
| `lam_k` / `lam_u` | `1e-3` / `1e-2` | Smoothness penalty / Laplacian regularization strength |
| `bilevel.engine` | `'implicit'` | `implicit` (exact hypergradient) or `reinforce` (PBCS-style) |
| `bilevel.iterative` | `True` | Iterative pruning schedule (ts/te cubic easing, from all features) |
| `bilevel.max_outer_iter` | `600` | Upper-level iterations |
| `bilevel.outer_lr` | `0.2` | Initial upper learning rate (cosine decay) |
| `bilevel.clip_grad` | `10.0` | Gradient-norm clipping |
| `refine` | `True` | Greedy swap refinement of the discretized mask |

Equivalent CLI flags: `--task --dataset --engine --k --max_outer_iter --outer_lr --K --clip_constant --upper_obj --no_refine --n_labeled --labeled_per_class --n_noisy --noise_std --lam_k --lam_u --r --seed --upper_data --no_vr --final_select --verbose`.

## 📁 Experimental Results

All numbers are reproduced with the released code (seed 0) under the paper protocol: **partial labeling** + **massive redundant dims** + **extreme noisy dims `N(100,100)`** on the four synthetic datasets of the paper (`data/data.md`). On all four studies S2MAM recovers **exactly the informative feature set** (recall = precision = **1.00**) and reaches **oracle-level test performance**:

| Dataset (task) | Layout | Selected features | Test performance (S2MAM vs. baselines) |
| --- | --- | --- | --- |
| Additive regression | 8 true + 92 redundant + 10 noisy, 50/200 labeled | `{0,…,7}` | MSE **0.596** = oracle (full-feature: 50.33) |
| Friedman regression | 5 true + 95 redundant + 10 noisy, 50/200 labeled | `{0,…,4}` | MSE **3.645** = oracle (full-feature: 26.91) |
| Additive classification | 2 true + 98 redundant + 10 noisy, 10 labels/class | `{0, 1}` | ACC **88.50%** = oracle (full-feature: 49.00%) |
| Two-moons classification | 2 true + 8 redundant + 10 noisy, 10 labels/class | `{0, 1}` | ACC **96.00%** = oracle (full-feature: 50.00%) |

Notes:

* Friedman contains the interaction term `10·sin(πx₁x₂)`; an additive lower model can only partially express it, which is why the end-to-end test suite uses a relaxed recall threshold (≥ 0.6) — the seed-0 run above still recovers all five informative dims thanks to the greedy swap refinement.
* The two-moons study uses the REINFORCE engine with a small iteration budget (`--max_outer_iter 40`), as each upper-level evaluation requires refitting the LapSVM lower model.

Interpretability: the fitted per-feature spline components recover the ground-truth additive functions (e.g., `f¹(u)=−2sin(2u)`, `f⁶(u)=5u`), providing component-wise interpretability of the decision mechanism.

## ✅ Tests

```bash
python -m pytest tests -q
# .............................  [100%]
# 29 passed
```

The suite covers: B-spline partition of unity, penalty PSD-ness, Laplacian properties, **closed-form solution verified against a direct ridge reference**, **hypergradient verified against central finite differences**, projection operator correctness, cardinality schedules, both engines, paper-faithful data generation, and end-to-end selection-quality assertions under partial labeling + noisy dims.

## ☑️ Todo

- [ ] Real-world benchmarks (UCI / ADNI) from the paper
- [ ] GPU-accelerated batched LOO evaluation for very high `p`
- [ ] Warm-start multi-restart upper solver
- [ ] Interactive visualization of learned additive components

## 🤝 Acknowledgements

The REINFORCE-style upper-level machinery is adapted from the engineering design of [Probabilistic Bilevel Coreset Selection](https://github.com/x-zho14/Probabilistic-Bilevel-Coreset-Selection) (PBCS); the semi-supervised kernel models (LapRLS/LapSVM) originate from this repository's initial release.

## 📖 Citation

If you find S2MAM useful for your research, please cite:

```bibtex
@misc{s2mam,
  title  = {S2MAM: Semi-supervised Meta Additive Models via Bilevel Feature Selection},
  author = {S2MAM Authors},
  year   = {2026},
  url    = {https://github.com/zxlml/S2MAM}
}
```

## 📄 License

This project is released under the [MIT License](LICENSE).
