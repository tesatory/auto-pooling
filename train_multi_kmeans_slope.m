function [W] = train_multi_kmeans_slope(train_data,hid_num,max_epoch,N_active_hidden)
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

A = ones(hid_num,data_dim, prec);
dA = zeros(size(A), prec);
vA = zeros(size(A), prec);
AD = 2;
AS = sum(A.^AD,2);

for epoch = 1:max_epoch
	if epoch > 5
		mom = 0.9;
	end
	
	order = randperm(data_sz);    
    for n = (1:data_sz / batchsz)
		% set visible from the train data
		chosen = order(1+batchsz*(n-1):batchsz*n);
		v = train_data(chosen,:)';

        dist_sq = bsxfun(@plus, A*(v.^2) -2*(A.*W)*v, sum(W.^2 .* A,2));
        [x,y] = sort(dist_sq);
        h = zeros(hid_num, size(v,2));
        for i = 1:size(v,2)
            h(y(1:N_active_hidden,i),i) = 1;
        end

        % positive weight update
        if epoch < 15
		dW = dW + A.*(h*v') - bsxfun(@times,A.*W,sum(h,2));
        vW = mom * vW + lrate * (dW / batchsz);
		W = W + vW;
		dW = zeros(size(W), prec);
        else
        dA = dA + 2*(W.*(h*v')) - h*(v.^2)' - bsxfun(@times,W.^2,sum(h,2));
        dA = dA - 1 * A;
        vA = mom * vA + lrate * (dA / batchsz);
		A = A + vA;
		dA = zeros(size(A), prec);
        A = A .* (A > 0); % keep positive
        A = A .* (A < 1) + 1 - (A < 1); % keep positive
%         A = bsxfun(@rdivide, A, (sum(A.^AD,2) ./ AS).^(1/AD));

        
        end
	end
	fprintf(1,'epoch %d\n', epoch);
	
	if mod(epoch,1) == 0
		% plot learning process in real-time
        subplot(1,3,1);display_network(W');title('W');
        subplot(1,3,2);display_network(A');title('A');
        subplot(1,3,3);display_network((A.*W)');title('A.*W');
		pause(0.1);
    end
end
