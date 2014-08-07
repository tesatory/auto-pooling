function [cost, grad] = cost_noisy_hidden_AE(theta, visibleSize, hiddenSize, ...
    lambda, data, hid_noise1, hid_noise2)
W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

datasz = size(data,2);

a1 = bsxfun(@plus, W1 * data, b1);
h1 = 1./(1 + exp(-a1));
h1 = h1 .* hid_noise1;
% h1 = h1 + hid_noise2;
a2 = bsxfun(@plus, W2 * h1, b2);
% h2 = 1./(1 + exp(-a2));
h2 = a2;

cost = mean(sum(0.5*(h2 - data).^2,1),2);
% cost = - mean(sum(data .* log(h2) + (1-data) .* log(1-h2),1),2);

% t2 = (h2 - data)  .* h2 .* (1 - h2) / datasz;
t2 = (h2 - data) / datasz;
b2grad = sum(t2,2);
W2grad = t2 * h1';

t1 = W2' * t2;
t1 = t1 .* h1 .* (1 - h1);
b1grad = sum(t1,2);
W1grad = t1 * data';

% add weight regularization cost
cost = cost + lambda*0.5*(sum(sum(W1.^2)) + sum(sum(W2.^2)));
W1grad = W1grad + lambda * W1;
W2grad = W2grad + lambda * W2;

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];
end