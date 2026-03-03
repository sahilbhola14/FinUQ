#ifndef DEFINITIONS_HPP
#define DEFINITIONS_HPP

/* #include "gamma.hpp" */

#define COMMENT_BLOCK(text) /* text */

/* type of distribution */
enum Distribution {Normal, ZeroOne, MinusOnePlusOne, PowTwo, Ones}; // N(0, 1); U[0, 1]; U[-1, 1]; U[2^k, 2^{k+1}]
/* precision models */
enum Precision {Double, Single, Half};
/* bound type */
enum BoundType {Deterministic, Hoeffding, Bernstein};
/* rounding error random variable model */
enum BoundModel {Uniform, Beta};

#endif
