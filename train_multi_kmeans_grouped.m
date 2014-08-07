function [W,A] = train_multi_kmeans_grouped(train_data, group_num, group_sz, max_epoch,A)
% meta parameters
lrate = 0.001; % for W and bv
batchsz = 100;

data_dim = size(train_data,2);
data_sz = size(train_data,1);

mom = 0;

useSingle = true;
prec = 'double';
if useSingle
    prec = 'single';
    train_data = single(train_data);
    lrate = single(lrate);
    mom = single(mom);
end
W = randn(group_sz,data_dim,group_num, prec) * 0.1;
% W = train_data(randi([1 data_sz],hid_num,1),:);
dW = zeros(size(W), prec);
vW = zeros(size(W), prec);

% A = (rand(data_dim, group_num, prec) * 2) / sqrt(group_num);
dA = zeros(size(A), prec);
vA = zeros(size(A), prec);
AD = 2;
A = A .* (A > 0); % keep positive
% A = bsxfun(@rdivide, A, (sum(A.^AD,2) ./ group_num).^(1/AD));
A = bsxfun(@rdivide, A, (sum(A.^AD,1) ./ data_dim).^(1/AD));

cost = zeros(group_num,max_epoch);

for epoch = 1:max_epoch
	if epoch > 5
		mom = 0.9;
	end
	
	order = randperm(data_sz);    

    for n = (1:data_sz / batchsz)
		% set visible from the train data
		chosen = order(1+batchsz*(n-1):batchsz*n);

        for g = 1:group_num
            v = train_data(chosen,:)';
            v = bsxfun(@times, v, A(:,g));
            dist_sq = bsxfun(@plus, bsxfun(@plus, -2*W(:,:,g)*v, ...
                sum(v.^2,1)), sum(W(:,:,g).^2,2));
            [x,y] = sort(dist_sq);
            cost(g,epoch) = cost(g,epoch) + sum(x(1,:));
            h = zeros(group_sz, size(v,2));
            for i = 1:size(v,2)
                h(y(1,i),i) = 1;
            end
            
            % positive weight update
            dW(:,:,g) = h*v' - bsxfun(@times,W(:,:,g),sum(h,2));
            dA(:,g) = 2*sum((W(:,:,g)'*h) .* v,2) - ...
                2*A(:,g).*sum(bsxfun(@times, v.^2, sum(h,1)),2);
        end
        vW = mom * vW + lrate * (dW / batchsz);
        W = W + vW;
        dW = zeros(size(W), prec);
        
        if epoch > 20
        vA = mom * vA + lrate * (dA / batchsz);
		A = A + vA;
		dA = zeros(size(A), prec);
        A = A .* (A > 0); % keep positive
%         A = bsxfun(@rdivide, A, (sum(A.^AD,2) ./ group_num).^(1/AD));      
        A = bsxfun(@rdivide, A, (sum(A.^AD,1) ./ data_dim).^(1/AD));
        end
    end
    cost(:,epoch) = cost(:,epoch) / data_sz;
    fprintf(1,'epoch %d cost %f \n', epoch, sum(cost(:,epoch)));
	
	if mod(epoch,1) == 0
		% plot learning process in real-time
        for g = 1:group_num
            subplot(2,group_num,g);display_network(W(:,:,g)');title('W');
        end
        subplot(2,group_num,1+group_num);display_network(A);title('A');
        subplot(2,group_num,2+group_num);plot(cost');legend('show');
		pause(0.1);
    end
end