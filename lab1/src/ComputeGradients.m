function [grad_b, grad_W] = ComputeGradients(X, Y, P, W, lambda)
    N = size(X,2);
    d = size(X,1);
    K = size(Y,1);

    % Initialize 
    Lb = zeros(K,1);
    LW = zeros(K,d);

    
    for i = 1:N
        x = X(:,i);
        y = Y(:,i);
        p = P(:,i);
        g = -(y-p);
        Lb = Lb + g;
        LW = LW + g*x';
    end

    
    grad_W = LW/N + (2*lambda*W);
    grad_b = Lb/N;
end

