function acc = ComputeAccuracy(X, y, W, b)
    N = size(X,2);
    P = EvaluateClassifier(X, W, b);
    
    [~, argmax] = max(P);
    acc = sum((argmax-1)==y)/N;
end