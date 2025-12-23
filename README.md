# A Variance-Informed Probabilistic Framework for Quantifying Computational Uncertainty Due to Finite-Precision Arithmetic

## Overview

Modern computer hardware increasingly supports low- and mixed-precision arithmetic to improve computational efficiency.
While these approaches offer substantial performance benefits, the introduce significant rounding erros that can significantly impact the reliability of numerical simulations and predictive models.
This project develops a variance-informed probabilistic rounding error analysis for quantifying the uncertainty due to finite-precision arithmetic.
The framework complements traditional sources of uncertainties and numerical errors, such as sampling uncertainty, parameteric uncertainty, and numerical discretization error, to enalbe reliable and efficient predictive modeling.

---

## Motivation

Classical deterministic rounding error analysis often yields overly conservative error bounds that scale poorly with problem size.
In contrast, probabilistic approaches exploit statistical structure in rounding errors to obtain tighter and more realistic estimates.
Building on this idea, we model rounding errors as **bounded, independent, and identically distributed random variables** and explicitly incorporate their **mean, variance, and bounds** into the analysis.

---

## Key Contributions

- **Variance-informed probabilistic rounding error analysis**
  - Introduces a new problem-size-dependent constant that depends on statistical properties of rounding errors.

- **Rigorous scaling results**
  - We prove that
   \[
    \hat{\gamma}_n \propto \sqrt{n}.
    \]
    using statistical arguments without ad-hoc assumptions.

- **Substantial improvement over deterministic bounds**
  - Rounding error estimates improve by **up to six orders of magnitude** for large arithmetic operations in low precision.

- **Unified uncertainty quantification**
  - Rounding uncertainty is quantified alongside discretization, sampling, and parameter uncertainties.
  - Enables principled **resource allocation** and **mixed-precision decision-making**.

---

## Numerical Experiments

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
├── include/          # Mathematical derivations and proofs
├── src/     # Numerical experiments and benchmarks
├── scripts/         # Reproducibility and experiment scripts
└── README.md
