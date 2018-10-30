function [bstar, Wstar] = MiniBatchGD(X, Y, W, b, lambda, eta)
    P = EvaluateClassifier(X, W, b);
    [grad_b, grad_W] = ComputeGradients(X, Y, P, W, lambda);
    
    % Equations 8 and 9
    bstar = b-eta*grad_b;
    Wstar = W-eta*grad_W;
end