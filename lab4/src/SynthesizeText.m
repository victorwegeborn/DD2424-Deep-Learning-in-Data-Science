function text = SynthesizeText(RNN, h, x, n)
   
    text = zeros(size(x,1), n); 
    for t = 1:n
        a = RNN.W * h + RNN.U * x + RNN.b;
        h = tanh(a);
        o = RNN.V * h + RNN.c;
        p = softmax(o);
        
        
        cp = cumsum(p);
        a = rand;
        ixs = find(cp-a >0);
        ii = ixs(1);
        
        x = zeros(size(x));
        x(ii) = 1;
        text(:,t) = x;
    end
end