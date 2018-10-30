classdef Network < handle
    properties
        HyperParams
        RNN
        MOMENTUM 
    end
    methods
        function obj = Network(HyperParams, RNN)
            obj.HyperParams = HyperParams;
            obj.RNN = RNN;
            obj.MOMENTUM.V = zeros(size(obj.RNN.V));
            obj.MOMENTUM.W = zeros(size(obj.RNN.W));
            obj.MOMENTUM.U = zeros(size(obj.RNN.U));
            obj.MOMENTUM.c = zeros(size(obj.RNN.c));
            obj.MOMENTUM.b = zeros(size(obj.RNN.b));
        end
        
        function [P, H] = Forward(obj, X, h0)
            P = zeros(obj.HyperParams.K, obj.HyperParams.seq_length);
            H = zeros(obj.HyperParams.m, obj.HyperParams.seq_length);
            
            ht = h0;
            
            for t = 1:obj.HyperParams.seq_length
                a = obj.RNN.W * ht + obj.RNN.U * X(:,t) + obj.RNN.b;
                ht = tanh(a);
                H(:,t) = ht;
                o = obj.RNN.V * H(:,t) + obj.RNN.c;
                P(:,t) = softmax(o);
            end
            
            H = [h0, H];
        end
        
        function GRADS = Backward(obj, X, Y, P, H)
            % set sizes of grads
            GRADS.V = zeros(size(obj.RNN.V));
            GRADS.W = zeros(size(obj.RNN.W));
            GRADS.U = zeros(size(obj.RNN.U));
            GRADS.c = zeros(size(obj.RNN.c));
            GRADS.b = zeros(size(obj.RNN.b));
            
            dL_do = -(Y-P)';
            
            GRADS.c = sum(dL_do,1)';            
            GRADS.V = dL_do' * H(:,2:end)';
            
            
            dL_dh = zeros(obj.HyperParams.seq_length, obj.HyperParams.m);
            dL_da = zeros(obj.HyperParams.seq_length, obj.HyperParams.m);
            
            dL_dh(end,:) = dL_do(end,:) * obj.RNN.V;            
            dL_da(end,:) = dL_dh(end,:) .* (1 - H(:,end).^2)';
            for t = obj.HyperParams.seq_length-1:-1:1
                dL_dh(t,:) = dL_do(t,:) * obj.RNN.V + dL_da(t+1,:) * obj.RNN.W;
                dL_da(t,:) = dL_dh(t,:) .* (1 - H(:,t+1).^2)';
            end
            
            GRADS.U = dL_da' * X';
            GRADS.b = sum(dL_da,1)';
            GRADS.W = dL_da' * H(:,1:end-1)';
            
            % clip gradient
            for f = fieldnames(GRADS)'
                GRADS.(f{1}) = max(min(GRADS.(f{1}), 5), -5);
            end
        end
        
        
        function loss = ComputeLoss(obj, X, Y, hprev)
            [P, ~] = obj.Forward(X, hprev);
            
            inner = sum(Y .* P, 1);
            loss = -sum(log(inner), 2);
        end
        
        
        function num_grads = ComputeGradsNum(obj, X, Y, h)
            for f = fieldnames(obj.RNN)'
                disp('Computing numerical gradient for')
                disp(['Field name: ' f{1} ]);
                num_grads.(f{1}) = obj.ComputeGradNumSlow(X, Y, f{1} , h);
            end
        end
        
        
        function grad = ComputeGradNumSlow(obj, X, Y, f, h)
            n = numel(obj.RNN.(f));
            grad = zeros(size(obj.RNN.(f)));
            hprev = zeros(size(obj.RNN.W, 1), 1);
            for i=1:n
                obj.RNN.(f)(i) = obj.RNN.(f)(i) - h;
                l1 = obj.ComputeLoss(X, Y, hprev);
                obj.RNN.(f)(i) = obj.RNN.(f)(i) + (2*h);
                l2 = obj.ComputeLoss(X, Y, hprev);
                obj.RNN.(f)(i) = obj.RNN.(f)(i) - h;
                grad(i) = (l2-l1)/(2*h);
            end
        end
        
        function text = SynthesizeText(obj, h, x, n)
            text = zeros(size(x,1), n); 
            for t = 1:n
                a = obj.RNN.W * h + obj.RNN.U * x + obj.RNN.b;
                h = tanh(a);
                o = obj.RNN.V * h + obj.RNN.c;
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
        
        
        function AdaGrad(obj, GRADS)
            for f = fieldnames(obj.RNN)'
                obj.MOMENTUM.(f{1}) = obj.MOMENTUM.(f{1}) + GRADS.(f{1}).^ 2;
                obj.RNN.(f{1}) = obj.RNN.(f{1}) - obj.HyperParams.eta*(GRADS.(f{1})./(obj.MOMENTUM.(f{1}) + 1e-8).^(0.5));
            end
        end
        
        
        function VanillaSGD(obj, GRADS)
            for f = fieldnames(obj.RNN)'
                obj.RNN.(f{1}) = obj.RNN.(f{1}) - obj.HyperParams.eta * GRADS.(f{1});
            end
        end
    end
end