function J = ComputeCost(X, Y, W, b, lambda)
    N = size(X,2);
    L2 = lambda*sumsqr(W);
    J = 0;
    P = EvaluateClassifier(X, W ,b);
    
    for i = 1:N
        y = Y(:,i);
        p = P(:,i);        
        J = J + (-log(y'*p));
    end 
    
    J = (J/N) + L2;
end