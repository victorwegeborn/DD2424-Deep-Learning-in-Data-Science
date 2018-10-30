function P = EvaluateClassifier(X, W, b) 
    K = size(W,1);
    N = size(X,2);
    P = zeros(K,N);
    
    for i = 1:N
        s = W*X(:,i) + b;
        P(:,i) = SoftMax(s);
    end 
end

% s is column vector
function p = SoftMax(s)
    p = exp(s)./sum(exp(s));
end