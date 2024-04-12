function prob = comp_thomas_prob_bernstein(system_size, precision, test_lambdas)
    %% Compute the higham probability for the test lambdas
    p_b_one     = comp_bernstein_prob(1, precision, test_lambdas);
    p_b_two     = comp_bernstein_prob(2, precision, test_lambdas);
    % LU Decomposition bound probability 
    q_lu    = 1 - 3*(1-p_b_one);
    t_lu    = 1 - (system_size-1)*(1-q_lu); 
    % Forward Substitution bound probability
    q_fs    = 1 - 2*(1-p_b_one);
    t_fs    = 1 - (system_size-1)*(1-q_fs);
    % Backward Substitution bound probability
    q_bs    = 1 - ((1-p_b_two) + (1-p_b_one));
    q_bsn   = 1 - (1-p_b_one);
    t_bs    = 1 - ((system_size-1)*(1-q_bs) + (1-q_bsn));
    % Total thomas probability
    prob = 1 - ( (1-t_lu) + (1-t_fs) + (1-t_bs) );
end
