function [cost,grad,h1] = sparseAutoencoderLinearCost(theta, visibleSize, hiddenSize, ...
                                                            lambda, sparsityParam, beta, data)
% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Copy sparseAutoencoderCost in sparseAutoencoderCost.m from your
%   earlier exercise onto this file, renaming the function to
%   sparseAutoencoderLinearCost, and changing the autoencoder to use a
%   linear decoder.
% -------------------- YOUR CODE HERE --------------------                                    


% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

a1 = bsxfun(@plus, W1 * data, b1);
h1 = 1./(1 + exp(-a1));
a2 = bsxfun(@plus, W2 * h1, b2);
h2 = a2;

cost = mean(sum(0.5*(h2 - data).^2,1),2);
cost_sparse = sparsityParam * log(sparsityParam ./ mean(h1,2)) + ...
    (1 - sparsityParam) * log((1-sparsityParam) ./ (1-mean(h1,2)));
cost = cost + beta * sum(cost_sparse);

datasz = size(data,2);
t2 = (h2 - data)/ datasz;
b2grad = sum(t2,2);
W2grad = t2 * h1';

t1 = W2' * t2;
t3 = beta * (- sparsityParam ./ mean(h1,2) + ...
    (1 - sparsityParam) ./ (1 - mean(h1,2))) / datasz;
t1 = bsxfun(@plus, t1, t3);
t1 = t1 .* h1 .* (1 - h1);
b1grad = sum(t1,2);
W1grad = t1 * data';

% add weight regularization cost
cost = cost + lambda*0.5*(sum(sum(W1.^2)) + sum(sum(W2.^2)));
W1grad = W1grad + lambda * W1;
W2grad = W2grad + lambda * W2;

%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

