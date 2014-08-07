function [W] = train_multi_kmeans_topo(train_data,hid_num,max_epoch,N_active_hidden,inhib_cost)
% meta parameters
lrate = 0.01; % for W and bv
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
W = randn(hid_num,data_dim, prec) * 0.1;
% W = train_data(randi([1 data_sz],hid_num,1),:);
dW = zeros(size(W), prec);
vW = zeros(size(W), prec);

for epoch = 1:max_epoch
	if epoch > 5
		mom = 0.9;
	end
	
	order = randperm(data_sz);    
    cost = 0;
    for n = (1:data_sz / batchsz)
		% set visible from the train data
		chosen = order(1+batchsz*(n-1):batchsz*n);
		v = train_data(chosen,:)';

        
        dist_sq = bsxfun(@plus, bsxfun(@plus, -2*W*v, sum(v.^2,1)), sum(W.^2,2));
        h = zeros(hid_num, size(v,2));
        for m = 1:N_active_hidden
            [x,y] = sort(dist_sq);
            cost = cost + sum(x(1,:));
            for i = 1:size(v,2)
                h(y(1,i),i) = 1;
                dist_sq(:,i) = dist_sq(:,i) + inhib_cost(:,y(1,i));
                dist_sq(y(1,i),i) = inf;
            end
        end
        
        % positive weight update
		dW = dW + (h*v' - bsxfun(@times,W,sum(h,2)));
        vW = mom * vW + lrate * (dW / batchsz);
		W = W + vW;
		dW = zeros(size(W), prec);
	end
	fprintf(1,'epoch %d cost %f\n', epoch, cost / data_sz);
	
	if mod(epoch,1) == 0
		% plot learning process in real-time
        subplot(1,2,1);display_network(W');
        subplot(1,2,2);display_network(h);

		pause(0.1);
    end
end
