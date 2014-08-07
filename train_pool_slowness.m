function pool = train_pool_slowness(hdataA,hdataB,max_epoch,lambda,alpha)
sz = size(hdataA,2)^0.5;
data_sz = size(hdataA,1);

% meta parameters
lrate = 0.1;
batchsz = data_sz;
moment = 0;

pool = zeros(sz^2,4);
% a = zeros(sz,sz);
% a(1:sz/2,1:sz/2) = 1;
% pool(:,1) = a(:);
% a = zeros(sz,sz);
% a(1:sz/2,sz/2+1:sz) = 1;
% pool(:,2) = a(:);
% a = zeros(sz,sz);
% a(sz/2+1:sz,1:sz/2) = 1;
% pool(:,3) = a(:);
% a = zeros(sz,sz);
% a(sz/2+1:sz,sz/2+1:sz) = 1;
% pool(:,4) = a(:);

for i = 1:size(pool,2)
    pool(randi(sz^2),i) = 1;
end
pool_speed = zeros(size(pool));
costs = zeros(max_epoch,1);

for epoch = 1:max_epoch    
%     order = randperm(data_sz);  % shuffle training data
    if epoch == 5
%         moment = 0.9;
    end
	for n = (1:data_sz / batchsz)
		% set visible from the train data
% 		chosen = order(1+batchsz*(n-1):batchsz*n);
        
%         [cost, grad] = cost_pool_variance(pool,1, lambda,alpha, hdataA(chosen,:), hdataB(chosen,:));
        [cost, grad] = cost_pool_variance(pool,1, lambda,alpha, [], []);
        costs(epoch) = mean(cost(:));
        
        pool_speed = pool_speed * moment - grad;        
        pool = pool + lrate * pool_speed;
        pool = pool .* (pool >= 0);
%         pool = bsxfun(@rdivide, pool, sum(pool,2));
%         pool = bsxfun(@rdivide, pool, sum(pool,1)) * 121;
    end
    fprintf('epoch %d cost %f sum %f \n', epoch, cost,mean(sum(pool,1),2));

    if mod(epoch,1) == 0
%         imagesc(reshape(pool,sz,sz));
%         display_network(pool);
view_data_gray(pool',sz,sz,4,1);
        drawnow
    end
end
