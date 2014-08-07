function [W,b] = get_trans_prob(data1, data2, useMinFunc,max_iter)
data_dim1 = size(data1,2);
data_dim2 = size(data2,2);
data_sz = size(data1,1);
lambda = 0.0;

W = randn(data_dim2,data_dim1)*0.01 + 1;
b = zeros(data_dim2,1);

% normilize
g = sum(W,2) + b;
W = bsxfun(@rdivide, W, g);
b = b ./ g;

if useMinFunc
    options = struct;
    options.Method = 'lbfgs';
    options.maxIter = 400;
    options.display = 'on';
    
    theta = [W(:); b(:)];
    [optTheta, cost] = minFunc( @(p) cost_trans_prob(p,lambda, data1',data2'), ...
        theta, options);
    
    W = reshape(optTheta(1:data_dim2*data_dim1), data_dim2, data_dim1);
    b = optTheta(data_dim1*data_dim2+1:data_dim2*data_dim1+data_dim2);
    fprintf('cost %f \n',cost);
else
    vW = zeros(size(W));
    vb = zeros(size(b));
    mom = 0;

    batch_sz = 100;
    lrate = 0.01;
    costs = zeros(max_iter,1);
    for t = 1:max_iter
        if t > 5
            mom = 0.9;
        end
        r = randperm(data_sz);
        
        theta = [W(:); b(:)];
        % [cost,grad] = cost_trans_prob(theta,lambda,data1',data2');
        [cost,grad] = cost_trans_prob(theta,lambda,data1(r(1:batch_sz),:)',data2(r(1:batch_sz),:)');
        
%         if 0
%             numgrad = computeNumericalGradient( @(x) cost_trans_prob(x, ...
%                 data1(r(1:batch_sz),:)',data2(r(1:batch_sz),:)'), theta);
%             diff = norm(numgrad-grad)/norm(numgrad+grad);
%             display(diff);
%             assert(diff < 1e-9);
%         end
        
        costs(t:end,1) = cost;
        W_grad = reshape(grad(1:data_dim1*data_dim2), data_dim2, data_dim1);
        b_grad = grad(data_dim1*data_dim2+1:data_dim1*data_dim2+data_dim2);
        
        vW = mom * vW - W_grad;
        W = W + lrate * vW;
        W = W .* (W > 0);
        vb = mom * vb - b_grad;
        b = b + lrate * vb;
        b = b .* (b > 0);
        
        % normilize
        W = (W + W')/2;
        g = sum(W,2) + b;
        W = bsxfun(@rdivide, W, g);
        b = b ./ g;
        
        if mod(t,1) == 0
            subplot(1,2,1);display_network(W.*(1-eye(size(W,1))));
            %         subplot(1,2,1);imagesc(W);
            subplot(1,2,2);plot(costs);
            fprintf('iter %d cost %f \n',t,cost);
            drawnow
        end
    end
end