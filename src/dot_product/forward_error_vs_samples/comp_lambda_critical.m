function lambda_critical = comp_lambda_critical(prob, lambda_vals, confidence_levels)
    lambda_critical = zeros(1, length(confidence_levels));

    for ii = 1:length(confidence_levels)
        confidence_level = confidence_levels(ii);
        lambda_critical(ii) = lambda_vals(find(prob >= confidence_level(ii), 1, 'first'));
    end

    assert (length(lambda_critical) == length(confidence_levels));
end