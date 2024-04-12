function abs_error = comp_abs_error(val_true, val_pred)
    % val_true: true value
    % val_pred: predicted value

    val_true_d = convert_precision(val_true, 'double');
    val_pred_d = convert_precision(val_pred, 'double');

    abs_error = abs(val_pred_d - val_true_d);

    assert(isa(abs_error, 'double'));
end