function [cost,grad] = cost_pool_ae(pool, X1, X2)
% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 

P = (X1 + X2) * pool;
Y = P * pool';

cost = mean(sum(0.5*(X1 - Y).^2,2),1) + ...
    mean(sum(0.5*(X2 - Y).^2,2),1);

grad = ((2*Y - X1 - X2)' * P + (X1 + X2)' * ((2*Y - X1 - X2) * pool)) / size(X1,1);
end

