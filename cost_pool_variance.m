function [cost, grad] = cost_pool_variance(pool, beta, lambda, alpha, hdataA, hdataB)
% r = randperm(size(hdataA,1));
% 
% d1 = hdataA - hdataB;
% d2 = hdataA - hdataB(r,:);
% 
% if beta == 0.5
%     epsilon = 0.001;
% else
%     epsilon = 0;
% end
% 
% e1 = sum((abs(d1 * pool) + epsilon).^beta,1);
% e2 = sum((abs(d2 * pool) + epsilon).^beta,1);
% cost = e1 ./ e2;
% 
% if beta == 0.5
%     grad = 0.5 * bsxfun(@times, d1' * (sign(d1 * pool) ./ (abs(d1 * pool) + epsilon).^0.5), e2) - ...
%         0.5 * bsxfun(@times, d2' * (sign(d2 * pool) ./ (abs(d2 * pool) + epsilon).^0.5), e1);
% elseif beta == 1
%     grad = bsxfun(@times, d1' * sign(d1 * pool), e2) - ...
%         bsxfun(@times, d2' * sign(d2 * pool), e1);
% elseif beta == 2
%     grad = 2 * bsxfun(@times, d1' * (d1 * pool), e2) - ...
%         2 * bsxfun(@times, d2' * (d2 * pool), e1);
% else
%     assert(false);
% end
% grad = bsxfun(@rdivide, grad, (e2.^2));


% overlapping cost
% cost = cost + lambda * sum(pool .* bsxfun(@minus, sum(pool,2), pool), 1);
grad = -ones(size(pool));
grad = grad + lambda * bsxfun(@minus, sum(pool,2), pool);
cost = 0;
% cost = cost - lambda * sum(pool(:));
% grad = grad - lambda;
% 
% % alpha = 0.01;
% cost  = cost + alpha * sum(sum(pool,2).^2);
% grad = bsxfun(@plus, grad, alpha * 2 * sum(pool,2));
