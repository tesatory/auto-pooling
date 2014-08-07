function [W1,W2,b1,b2] = train_sparseAE_linear(train_data,hiddenSize,sparsityParam,beta,maxIter,theta)
visibleSize = size(train_data,2);
% sparsityParam = 0.01; % desired average activation of the hidden units.
lambda = 3e-3;         % weight decay parameter       
% beta = 5;              % weight of sparsity penalty term   

assert(nargin == 5 || nargin == 6);
if nargin == 5
    theta = initializeParameters(hiddenSize, visibleSize);
end

options = struct;
options.Method = 'lbfgs'; 
options.maxIter = maxIter;
options.display = 'on';

[optTheta, cost] = minFunc( @(p) sparseAutoencoderLinearCost(p, ...
                                   visibleSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, train_data'), ...
                              theta, options);
                          
W1 = reshape(optTheta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(optTheta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = optTheta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = optTheta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% display_network(W1');