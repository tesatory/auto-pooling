function [cost,grad] = denoisingAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, noise_rate, data, noise)
assert(nargin == 6 || nargin == 7)
if nargin == 6
    noise = (rand(size(data)) > noise_rate);
end
data_noisy = data .* noise;

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);


a1 = bsxfun(@plus, W1 * data_noisy, b1);
h1 = 1./(1 + exp(-a1));
a2 = bsxfun(@plus, W2 * h1, b2);
h2 = 1./(1 + exp(-a2));

cost = mean(sum(0.5*(h2 - data).^2,1),2);

% cost_sparse = sparsityParam * log(sparsityParam ./ mean(h1,2)) + ...
%     (1 - sparsityParam) * log((1-sparsityParam) ./ (1-mean(h1,2)));
% cost = cost + beta * sum(cost_sparse);

datasz = size(data,2);

t2 = (h2 - data)  .* h2 .* (1 - h2) / datasz;
b2grad = sum(t2,2);
W2grad = t2 * h1';

t1 = W2' * t2;
% t3 = beta * (- sparsityParam ./ mean(h1,2) + ...
%     (1 - sparsityParam) ./ (1 - mean(h1,2))) / datasz;
% t1 = bsxfun(@plus, t1, t3);
t1 = t1 .* h1 .* (1 - h1);
b1grad = sum(t1,2);
W1grad = t1 * data_noisy';

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

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

