function unit_rf = comp_urd(precision)
        if precision == 'double'
            unit_rf = eps('double');
            unit_rf = unit_rf / 2;
        elseif precision == 'single'
            unit_rf = double(eps('single'));
            unit_rf = unit_rf / 2;
        elseif precision == 'half'
            unit_rf = double(eps('half'));
            unit_rf = unit_rf / 2;
        else
            error('Invalid precision');
        end
end
