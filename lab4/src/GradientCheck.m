

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


