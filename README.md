<h1 align="center">FinUQ</h1>
<h3 align="center">Beyond Zero-Mean Assumptions: Variance-Informed Probabilistic Rounding Error Analysis</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2404.12556">arXiv</a> |
  <a href="#overview">Overview</a> |
  <a href="#key-contributions">Key Contributions</a> |
  <a href="#project-layout">Project Layout</a> |
  <a href="#build-and-run">Build and Run</a> |
  <a href="#experiments">Experiments</a>
</p>

---

## Overview

`FinUQ` develops a probabilistic framework for quantifying floating-point rounding uncertainty in low- and mixed-precision computations.

Classical deterministic rounding-error bounds are often conservative and can scale poorly with operation count. This project instead models rounding errors as bounded random variables and uses distribution moments to derive tighter, confidence-calibrated uncertainty bounds that remain practical at scale.

The framework is designed to complement other uncertainty sources in scientific computing, including:

- sampling uncertainty,
- parametric uncertainty, and
- discretization error.

## Key Contributions

1. **Explicit confidence-calibrated probabilistic bounds**
   - Derives a corollary of Theorem 2.4 of Higham and Mary that recovers the $\sqrt{n}$ growth in $\tilde{\gamma}_n$.
   - Provides a closed-form confidence parameter $\lambda$ in terms of unit roundoff and target confidence.
   - Recovers the scaling $\lambda \propto (1-u)^{-1}$.

2. **Variance-informed analysis beyond zero-mean assumptions**
   - Introduces an operation-count-dependent constant $\hat{\gamma}_n$ that incorporates both first and second moments of rounding-error random variables.
   - Supports flexible uncertainty quantification in the presence of systematic bias.

3. **Moment-driven control of accumulation growth**
   - Shows that accumulation growth depends on how the rounding-error distribution is characterized, not only on stochastic assumptions.
   - Uses a log-domain bias model to systematically control growth in $\hat{\gamma}_n$.

4. **GPU-scale low-precision validation**
   - Validates bounds with CUDA experiments in `float` and `half` precision.
   - Covers dot products, sparse matrix-vector products (SuiteSparse matrices), and stochastic ODE settings where floating-point uncertainty interacts with other error sources.

## Project Layout

```text
.
├── CMakeLists.txt                # Build configuration
├── include/
│   ├── cpp/                      # C++ headers (theory + experiment interfaces)
│   └── cuda/                     # CUDA headers and kernels
├── src/
│   ├── cpp/                      # C++ experiment drivers and models
│   └── cuda/                     # CUDA implementations
├── scripts/
│   ├── dot_product/              # Dot-product postprocessing and plots
│   ├── matrix_market/            # Sparse matvec experiments and plots
│   ├── bvp/                      # ODE/BVP experiment utilities and plots
│   ├── compare_gamma/            # Gamma comparison scripts
│   ├── model_verification.py
│   └── rounding_error_model.py
└── README.md
```

## Build and Run

### Prerequisites

- CMake >= 3.10
- CUDA toolkit (configured for CUDA 12.1 in `CMakeLists.txt`)
- C++14 compiler
- Eigen3

### Build

```bash
cmake -S . -B build
cmake --build build -j
```

This generates the executable `run` in `build/`.

### Run Experiments

```bash
./build/run compare_gamma
./build/run dot_product
./build/run matrix_market
./build/run ode
```

If no argument is provided, `main.cpp` defaults to the internal `testing` path.

## Experiments

The repository includes numerical studies for:

- random dot products,
- sparse matrix-vector products,
- ODE/BVP-style stochastic settings, and
- comparison of probabilistic growth constants.

The experiments show that probabilistic bounds can remain tight in low precision while preserving predictive reliability.

## Reference

- ArXiv preprint: https://arxiv.org/abs/2404.12556
