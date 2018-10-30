
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

