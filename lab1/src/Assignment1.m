
addpath(genpath('/Users/victorwegeborn/Documents/KTH/vt18/dl/lab1/datasets/'));
clear;


% Task 1: Load data into matlab.
Train = LoadBatch('data_batch_1.mat');
Valid = LoadBatch('data_batch_2.mat');
Test  = LoadBatch('test_batch.mat');

% Task 2: Initialize W and b
K = 10;
N = 10000;
d = 3072;

rng(400);
W = 0.01*randn(K,d);
b = 0.01*randn(K,1);


%Evaluation of the gradients calculation
%{
lambda = 0;
batch  = 100;
[ngrad_b, ngrad_W] = ComputeGradsNumSlow(Train.X(:, 1:batch), Train.Y(:, 1:batch), W, b, lambda, 1e-6);

P = EvaluateClassifier(Train.X(:, 1:batch), W, b);
[agrad_b, agrad_W] = ComputeGradients(Train.X(:, 1:batch), Train.Y(:, 1:batch), P, W, lambda);

relativeError_b = abs(ngrad_b - agrad_b)./max(0, abs(ngrad_b)+abs(agrad_b));
relativeError_W = abs(ngrad_W - agrad_W)./max(0, abs(ngrad_W)+abs(agrad_W));

max(relativeError_b)
max(max(relativeError_W))
%}

eta      = 0.01; % Learning rate
lambda   = 1;    % Regularization
n_batch  = 100;  % Number of batches
n_epochs = 40;   % Number or epochs

% placeholders for plotting
Train_J = zeros(1,n_epochs);
Valid_J = zeros(1,n_epochs);


% Learning
for i = 1:n_epochs
    
    for j = 1:N/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end;
        Xbatch = Train.X(:, inds);
        Ybatch = Train.Y(:, inds);
        
        [b,W] = MiniBatchGD(Xbatch, Ybatch, W, b, lambda, eta);
    end
    
    Train_J(i) = ComputeCost(Train.X, Train.Y, W, b, lambda);
    Valid_J(i) = ComputeCost(Valid.X, Valid.Y, W, b, lambda);
end

acc = ComputeAccuracy(Test.X, Test.y, W, b);


xlabel('epochs')
ylabel('loss')
plot(1:n_epochs, Train_J, 1:n_epochs, Valid_J)
legend('Training loss','Validation loss')

showClassTemplate(W);

% ------- Helper Function definitions -------
function struct = LoadBatch(filename)
    D = load(filename);
    X = im2double(D.data');%double(rdivide(D.data', 255));
    y = D.labels';  
    
    Y = zeros(10, size(X,2));
    %y = double(D.labels') + 1;
    
    for i = 0:9 
        r = y == i;
        Y(i+1,r) = 1;
    end
    
    
    struct.X = X;
    struct.Y = Y;
    struct.y = y;
end


function showClassTemplate(W) 
    m = [];
    for i=1:10
        im = reshape(W(i, :), 32, 32, 3);
        s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:))); 
        s_im{i} = permute(s_im{i}, [2,1,3]);   
        m = [m s_im{i}];
    end
    montage(m);
end

