function [cost,grad] = cost_pool_ae2(pool, X1, X2, lambda)
% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 

P1 = X1 * pool;
P2 = X2 * pool;
Y1 = P1 * pool';
Y2 = P2 * pool';

cost = mean(0.5*sum((X1 - Y1).^2,2) + 0.5*sum((X2 - Y2).^2,2) + ...
    lambda*0.5*sum((P1 - P2).^2,2),1);

grad = (Y1 - X1)' * P1 + X1' * ((Y1 - X1) * pool) + ...
    (Y2 - X2)' * P2 + X2' * ((Y2 - X2) * pool) + ...
    lambda * (X1 - X2)' * (P1 - P2);

grad = grad / size(X1,1);
end

