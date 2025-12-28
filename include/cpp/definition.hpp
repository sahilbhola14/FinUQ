#ifndef DEFINITIONS_HPP
#define DEFINITIONS_HPP

#define COMMENT_BLOCK(text) /* text */

enum Distribution {Normal, ZeroOne, MinusOnePlusOne, PowTwo, Ones}; // N(0, 1); U[0, 1]; U[-1, 1]; U[2^k, 2^{k+1}]
enum Precision {Double, Single, Half};
enum BoundType {Deterministic, Hoeffding, Bernstein};
enum BoundModel {Uniform, Beta};

#endif
