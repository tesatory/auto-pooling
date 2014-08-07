function [pool, costs] = train_pool_ae2(X1,X2,hiddenSize,lambda,max_epoch, useSingle, useGPU, pool)
visibleSize = size(X1,2);
data_sz = size(X1,1);

% meta parameters
lrate = 0.01;
batchsz = 100;
if useSingle
    bigbatchsz = 5000;
else
    bigbatchsz = 5000;
end
moment = 0;

assert(nargin == 8 || nargin == 7);
if nargin == 7
%     pool = rand(visibleSize, hiddenSize) * 0.01;
%     pool = bsxfun(@rdivide, pool, (sum(pool.^2,1)).^0.5);
%     pool = pool / sqrt(visibleSize);

    pool = zeros(visibleSize,hiddenSize);
    r = randperm(visibleSize);
    for h=1:hiddenSize
        pool(r(h),h) = 1;
    end
else
%     moment = 0.9;   % middle of training
end

speed = zeros(size(pool));
% costs = zeros(data_sz / batchsz,max_epoch);

if useSingle
    X1 = single(X1);
    X2 = single(X2);
    pool = single(pool);
    speed = single(speed);
%     costs = single(costs);
end

if useGPU
    pool = gpuArray(pool);
    speed = gpuArray(speed);
%     costs = gpuArray(costs);
end

tic;
for epoch = 1:max_epoch    
    if epoch == 5
        moment = 0.9;
    end
    for nn = (1:data_sz / bigbatchsz)
        if useGPU
            clear X1gpu X2gpu
            X1gpu = gpuArray(X1(1+bigbatchsz*(nn-1):bigbatchsz*nn,:));
            X2gpu = gpuArray(X2(1+bigbatchsz*(nn-1):bigbatchsz*nn,:));
        else
            X1gpu = X1(1+bigbatchsz*(nn-1):bigbatchsz*nn,:);
            X2gpu = X2(1+bigbatchsz*(nn-1):bigbatchsz*nn,:);
        end
        for n = (1:bigbatchsz / batchsz)
            order = randperm(bigbatchsz);  % shuffle training data
            % set visible from the train data
            chosen = order(1+batchsz*(n-1):batchsz*n);
            X1_sub = X1gpu(chosen,:);
            X2_sub = X2gpu(chosen,:);
            
            if lambda == -1
                [cost, grad] = cost_pool_ae(pool, X1_sub, X2_sub);
            else
                [cost, grad] = cost_pool_ae2(pool, X1_sub, X2_sub, lambda);
            end
%             costs(n,epoch) = cost;
            
            speed = speed * moment - grad;
            pool = pool + lrate * speed;
            pool = pool .* (pool > 0); % keep positive            
        end
    end
    fprintf(1,'epoch %d - %d\r',epoch, n);
%     fprintf(1,'epoch %d - %d cost %f\r',epoch, n, mean(costs(:,epoch)));
    
%     if mod(epoch,1) == 0 && isdeployed == false
%         subplot(1,2,1); plot(mean(costs,1));
%         subplot(1,2,2);
        %         display_network(pool);
        %         imagesc(pool);
%         hist(pool(:));
%         drawnow
%     end
end
costs = 0;
toc;
