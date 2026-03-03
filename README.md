<h1 align="center">FinUQ</h1>
<h3 align="center">Beyond Zero-mean Assumptions: Variance-informed Probabilistic Rounding Error Analysis</h3>
<p align="center">
  <a href="https://arxiv.org/abs/2404.12556">📄 arXiv</a> |
  <a href="#Overview">Overview</a> |
  <a href="#Contributions">Contributions</a> |
  <a href="#Experiments">Experiments</a> |
  <a href="#Repository-Structure">Repository Structure</a> |
</p>

---

## Overview

Modern computer hardware increasingly supports low- and mixed-precision arithmetic to improve computational efficiency.
While these approaches offer substantial performance benefits, the introduce significant rounding erros that can significantly impact the reliability of numerical simulations and predictive models.
This project develops a variance-informed probabilistic rounding error analysis for quantifying the uncertainty due to finite-precision arithmetic.
The framework complements traditional sources of uncertainties and numerical errors, such as sampling uncertainty, parameteric uncertainty, and numerical discretization error, to enalbe reliable and efficient predictive modeling.

### Motivation

Classical deterministic rounding error analysis often yields overly conservative error bounds that scale poorly with problem size.
In contrast, probabilistic approaches exploit statistical structure in rounding errors to obtain tighter and more realistic estimates.
Building on this idea, we model rounding errors as **bounded, independent, and identically distributed random variables** and explicitly incorporate their **mean, variance, and bounds** into the analysis.

---

## Theoretical Contributions

This work highlights the following core contributions:

1. **Explicit and confidence-calibrated probabilistic bounds**
   - Derives a corollary of Theorem 2.4 of Higham and Mary that rigorously recovers the $\sqrt{n}$ growth in $\tilde{\gamma}_n$.
   - Provides a closed-form expression for the confidence parameter $\lambda$, explicitly in terms of unit roundoff and target confidence.
   - Recovers the scaling $\lambda \propto (1-u)^{-1}$, consistent with empirical findings in prior work.

2. **Variance-informed probabilistic rounding error analysis**
   - Introduces a new operation-count-dependent constant, $\hat{\gamma}_n$, that incorporates both first and second moments of the rounding error random variable.
   - Enables sharper and more flexible quantification of floating-point uncertainty beyond zero-mean assumptions.

3. **Moment-driven control of accumulation growth**
   - Shows that growth of operation-count-dependent constants is driven by how the rounding-error distribution is characterized, not only by stochastic assumptions.
   - Models bias directly in the log-domain to systematically control growth of $\hat{\gamma}_n$.

---

## Experiments

The framework is validated on the following problems:

- Random vector dot products
- Matrix-vector multiplication
- Linear system solvers
- Stochastic boundary value problems

The results demonstrate that probabilistic bounds remain tight at scale and allow aggressive use of low-precision arithmetic without sacrificing predictive accuracy.

---

## Repository Structure

```text
.
├── include/            # Mathematical derivations and proofs
├── src/                # Numerical experiments and benchmarks
├── lib/                # External libraries
├── scripts/            # Reproducibility and experiment scripts
└── README.md
