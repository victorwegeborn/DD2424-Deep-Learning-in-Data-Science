
% Assignment4.m

manager = DataManager('goblet_book.txt');


% setup hyper-parameters
HyperParams.eta = 0.09  ;
HyperParams.seq_length = 25;
HyperParams.K = manager.K;
HyperParams.m = 100;
HyperParams.epochs = 20;

% setup RNN parameters
sig = 0.01;
RNN.b = zeros(HyperParams.m, 1);
RNN.c = zeros(HyperParams.K, 1);
RNN.U = randn(HyperParams.m, HyperParams.K) * sig;
RNN.W = randn(HyperParams.m, HyperParams.m) * sig;
RNN.V = randn(HyperParams.K, HyperParams.m) * sig;


% Train network
network = Network(HyperParams, RNN);
smooth_loss = 0;
iter = 1;
smooth_loss_array = [];
result_file = fopen('results/task2.txt','w');
for i = 1:HyperParams.epochs
    hprev = zeros(HyperParams.m, 1);
    e = 1;
    fprintf(2, '\n\n epoch: %d \n\n', i)
    while e < length(manager.book_data) - HyperParams.seq_length
        [~, ~, X, Y] = manager.one_hot_batch(e, HyperParams.seq_length);
        
        [P, H] = network.Forward(X, hprev);   
        loss = network.ComputeLoss(X, Y, hprev);  
        GRADS = network.Backward(X, Y, P, H);
        network.AdaGrad(GRADS);
        %network.VanillaSGD(GRADS);
        hprev = H(:,end);
       
        
        if e == 1 && i == 1
            smooth_loss = loss;
            storer.initialize_loss(smooth_loss);
        end
        
        smooth_loss = 0.999 * smooth_loss + 0.001 * loss;
        smooth_loss_array = [smooth_loss_array, smooth_loss];
        
        if iter == 1 || mod(iter, 1000) == 0
            fprintf('iter: %d, ', iter);
            fprintf('Loss: %f \n', smooth_loss)
            text = network.SynthesizeText(hprev, X(:,1), 200);
            manager.PrintText(text);
            fprintf('\n\n')
        end
        
        if iter == 1 || mod(iter, 10000) == 0
            fprintf(result_file ,'\n\niter: %d, ', iter);
            fprintf(result_file, 'loss: %f \n', smooth_loss);
            text = network.SynthesizeText(hprev, X(:,1), 200);
            manager.WriteText(text, result_file);
        end
        
        storer.StoreOnLowest(smooth_loss, network.RNN);
        
        e = e + HyperParams.seq_length;
        iter = iter + 1;
    end

end

fclose(result_file);
text = network.SynthesizeText(hprev, X(:,1), 1000);
            manager.PrintText(text);
plot(1 : length(smooth_loss_array), smooth_loss_array)






% DataManager.m
classdef DataManager
    properties
        book_data
        book_chars
        char_to_index = containers.Map('KeyType', 'char', 'ValueType', 'int32')
        index_to_char = containers.Map('KeyType', 'int32', 'ValueType', 'char')
        K 
    end
    methods
        function obj = DataManager(filename)
            book_fname = strcat('data/', filename);
            fid = fopen(book_fname, 'r');
            obj.book_data = fscanf(fid, '%c');
            fclose(fid);
            
            obj.book_chars = unique(obj.book_data);
            obj.K = length(obj.book_chars);
            
            for i = 1:obj.K
                obj.char_to_index(obj.book_chars(i)) = i;
                obj.index_to_char(i) = obj.book_chars(i);
            end
        end
        
        function [X_chars, Y_chars, X, Y] = one_hot_batch(obj, index, seq_length)
            start = index;
            stop = start + seq_length;
            X_chars = obj.book_data(start:stop);
            Y_chars = obj.book_data(start+1:stop+1);
            
            X = zeros(obj.K, stop - start);
            Y = zeros(obj.K, stop - start);
            
            for i = 1:seq_length
                X(obj.char_to_index(X_chars(i)),i) = 1;
                Y(obj.char_to_index(Y_chars(i)),i) = 1;
            end
        end
        
        function PrintText(obj, text)
            for i = 1:size(text,2)
                [~, argmax] = max(text(:,i));

                string(i) = obj.index_to_char(int32(argmax)); %#ok<AGROW>
            end
            disp(string)
        end
        
        function WriteText(obj, text, file)
            for i = 1:size(text,2)
                [~, argmax] = max(text(:,i));

                string(i) = obj.index_to_char(int32(argmax)); %#ok<AGROW>
            end
            fprintf(file, string, '\n\n');
        end
    end
end


% LoadData.m
function [book_data, char_to_index, index_to_char, K] = LoadData(filename)
    book_fname = strcat('data/', filename);
    fid = fopen(book_fname, 'r');
    book_data = fscanf(fid, '%c');
    fclose(fid);
    
    book_chars = unique(book_data);
    K = length(book_chars);
    
    char_to_index = containers.Map(book_chars, 1:K);
    index_to_char = containers.Map(1:K, book_chars);
end


% HyperParams.m
classdef HyperParams
    properties
        K
        m
        seq_length 
        eta
        epochs
    end
end



% Network.m
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



% GradientCheck.m


manager = DataManager('goblet_book.txt');

% setup hyper-parameters
HyperParams.eta = 0.1;
HyperParams.seq_length = 25;
HyperParams.K = manager.K;
HyperParams.m = 5;


% setup RNN parameters
sig = 0.01;
RNN.b = zeros(HyperParams.m, 1);
RNN.c = zeros(HyperParams.K, 1);
RNN.U = randn(HyperParams.m, HyperParams.K) * sig;
RNN.W = randn(HyperParams.m, HyperParams.m) * sig;
RNN.V = randn(HyperParams.K, HyperParams.m) * sig;

[X_chars, Y_chars, X, Y] = manager.one_hot_batch(1, HyperParams.seq_length);

h0 = zeros(HyperParams.m, 1);
network = Network(HyperParams, RNN);
[P, H] = network.Forward(X, h0);
GRADS = network.Backward(X, Y, P, H);

N_GRADS = network.ComputeGradsNum(X, Y, 1e-4);


for f = fieldnames(network.RNN)'
    error = abs(GRADS.(f{1}) - N_GRADS.(f{1}));
    disp(['Max value for ' f{1}]);
    disp(max(max(error)))
    disp(['Mean value for ' f{1}]);
    disp(mean(mean(error)))
end


% SynthesizeText.m
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



% Storage.m
classdef Storage < handle
    properties 
        RNN_copy
        loss
    end
    methods
        function obj = Storage(RNN)
            obj.RNN_copy.b = zeros(size(RNN.b));
            obj.RNN_copy.W = zeros(size(RNN.W));
            obj.RNN_copy.V = zeros(size(RNN.V));
            obj.RNN_copy.c = zeros(size(RNN.c));
            obj.RNN_copy.U = zeros(size(RNN.U));
        end
        
        function initialize_loss(obj, smooth_loss)
            obj.loss = smooth_loss;
        end
        
        function StoreOnLowest(obj, smooth_loss, RNN)
            if smooth_loss < obj.loss
                obj.loss = smooth_loss;
                obj.RNN_copy = RNN;
            end
        end
        
        function WriteToDisc(obj)
            obj.RNN_copy;
            save('results/RNN_copy', 'obj.RNN_copy' ,'-mat');
        end
        
        function RNN = LoadFromDisc(obj)
            obj.RNN_copy = load('results/RNN_copy', '-mat');
            RNN = obj.RNN_copy;
        end
    end
end


