function [W,bv,bh] = train_sparseAE(train_data,hid_num,max_epoch,alpha)
% train a sparse AutoEncoder
data_dim = size(train_data,2);
data_sz = size(train_data,1);

% meta parameters
lrate = 0.01;
batchsz = 100;

% weights
bv = zeros(data_dim,1);
bh = zeros(hid_num,1);
l = 4 * sqrt(6 / (hid_num + data_dim));
W = rand(hid_num,data_dim) * 2 * l - l;
W = W / 10;
W_prime = W';

prec = 'double';
vW = zeros(size(W), prec);
vW_prime = zeros(size(W_prime), prec);
vbv = zeros(size(bv), prec);
vbh = zeros(size(bh), prec);
mom = 0;

binary = false;
weight_tied = true;

for epoch = 1:max_epoch
	fprintf(1,'epoch %d\r',epoch); 
    if epoch > 5
		mom = 0.9;
	end
	order = randperm(data_sz);  % shuffle training data
	for n = (1:data_sz / batchsz)
		% set visible from the train data
		chosen = order(1+batchsz*(n-1):batchsz*n);
		v = train_data(chosen,:)';

		% calc hidden values
		h = W*v + repmat(bh,1,batchsz);
 		h = 1 ./ (1 + exp(-h));  % sigmoid
        h = h .* (rand(size(h)) >= alpha);
        hh = h;
        
		% calc visible values
        rv = W_prime*h + repmat(bv,1,batchsz);
        if binary
			rv = 1 ./ (1 + exp(-rv));  % sigmoid
		end
		
% 		% calc cost
% 		if binary
% 			L = - sum(v .* log(rv) + (1-v) .* log(1-rv));
% 		else
% 			L = mean(sum((v - rv).^2,1));
% 		end
% 		cost(c) = mean(L);
% 		c = c+1;

		dbv = sum(v - rv, 2);
        t1 = (W_prime' * (v - rv)) .* h .* (1-h);
        dW_prime = (v - rv) * h';
        dbh = sum(t1,2);
        dW = t1 * v';
        
        vW_prime = mom * vW_prime + lrate * (dW_prime / batchsz);
		vW = mom * vW + lrate * (dW / batchsz);
		vbv = mom * vbv + lrate * (dbv / batchsz);
		vbh = mom * vbh + lrate * (dbh / batchsz);
		
		W_prime = W_prime + vW_prime;
		W = W + vW;
		bv = bv + vbv;
		bh = bh + vbh;
        
        if weight_tied
            W = (W + W_prime')/2;
            W_prime = W';
        end
	end
	
% 	subplot(2,3,1);plot(log(cost));
	subplot(2,3,3);hist(W(:),100);
% 	subplot(2,3,3);hist(W_prime(:),100);
	subplot(2,3,1);view_data(v',6,6,10,10);
	subplot(2,3,2);view_data(rv',6,6,10,10);
	subplot(2,3,4);view_data(W,6,6,10,10);
	subplot(2,3,5);view_data(W_prime',6,6,10,10);
	subplot(2,3,6);imagesc(hh);

	pause(0.01);		
end
