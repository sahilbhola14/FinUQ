#ifndef ODE_CUH
#define ODE_CUH

void launchStochasticODEExperiment(int N_lower, int N_param_samples,
                                   int bit_shift, int max_shift,
                                   int num_exps = 1, double confidence = 0.99);
#endif
