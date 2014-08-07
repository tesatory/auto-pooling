function [cost, grad] = cost_trans_prob(theta, lambda, dataX, dataY)
data_dim1 = size(dataX,1);
data_dim2 = size(dataY,1);
data_sz = size(dataX,2);

W = reshape(theta(1:data_dim2*data_dim1), data_dim2, data_dim1);
b = theta(data_dim1*data_dim2+1:data_dim2*data_dim1+data_dim2);

Y = bsxfun(@plus,W*dataX,b);
% Y = 1 ./  (1 + exp(-Y));

cost = - mean(sum(dataY .* log(Y) + (1-dataY) .* log(1-Y),1),2); 
% A = Y - dataY;
A = (Y - dataY) ./ Y ./ (1-Y);
W_grad = A*dataX'/ data_sz;
b_grad = mean(A,2);

% cost = mean(sum((Y - dataY).^2,1),2);
% W_grad = 2 * (Y - dataY)*dataX'/ data_sz;
% b_grad = 2 * mean(Y - dataY,2);

% add weight regularization cost
cost = cost + lambda*0.5*(sum(sum(W.^2)));
W_grad = W_grad + lambda * W;
% cost = cost + lambda*0.5*(sum(sum((W.*(1-eye(data_dim1))).^2)));
% W_grad = W_grad + lambda * (W.*(1-eye(data_dim1)));

grad = [W_grad(:) ; b_grad(:)];
end