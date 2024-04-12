function rel_error = comp_rel_error(val_true, val_pred)
    % val_true: true value
    % val_pred: predicted value

    val_true_d = convert_precision(val_true, 'double');
    val_pred_d = convert_precision(val_pred, 'double');

    rel_error = abs(val_pred_d - val_true_d) ./ abs(val_true_d);

    assert(isa(rel_error, 'double'));
end