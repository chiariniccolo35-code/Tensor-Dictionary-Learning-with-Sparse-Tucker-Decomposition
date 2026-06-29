function check_tensor_operations(Y, X, A0, B0, C0, itrs_max)
    % Inside-function calculation (no iteration for X_cb)
    X_cb_in = ttm(X, {C0, B0}, [3, 2]);
    X_cb_in = unfold(X_cb_in, 1);
    S_in = ttm(X, {A0, B0, C0});
    norm_in = norm(unfold(S_in, 1) - A0 * X_cb_in);
    
    % Simulated iterative process for A, B, C
    A = A0;
    B = B0;
    C = C0;
    itrs = 0;
    res = 1;
    
    while itrs < itrs_max && res > 1e-6
        S_in = ttm(X, {A, B, C});
        res = norm(unfold(S_in, 1) - A * X_cb_in);
        
        % Example update (depends on actual logic in GradTensor)
        grad_A = (unfold(Y, 1) - A * X_cb_in) * pinv(X_cb_in);
        A = A + 0.01 * grad_A;
        
        itrs = itrs + 1;
    end
    
    % Outside-function calculation (no iterations)
    X_cb_out = ttm(X, {C0, B0}, [3, 2]);
    X_cb_out = unfold(X_cb_out, 1);
    S_out = ttm(X, {A0, B0, C0});
    norm_out = norm(unfold(S_out, 1) - A0 * X_cb_out);
    
    % Display results
    fprintf('Norm inside function (initial): %f\n', norm_in);
    fprintf('Norm after iteration: %f\n', res);
    fprintf('Norm outside function: %f\n', norm_out);
end