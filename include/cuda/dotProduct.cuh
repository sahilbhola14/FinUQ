#ifndef DOTPRODUCT_CUH
#define DOTPRODUCT_CUH

// Kernels
void launchDotProductExperiment(int N_lower, int bit_shift = 2,
                                int max_shift = 12, int num_exps = 1,
                                double confidence = 0.99);
#endif
