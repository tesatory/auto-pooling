function [W,bv,bh] = train_RBM(train_data,hid_num,max_epoch)
% meta parameters
lrate = 0.01;

batchsz = 100;
CDk = 1;  % the number of gibbs sampling in contrastive divergence

data_dim = size(train_data,2);
data_sz = size(train_data,1);

mom = 0;

% sparse_cost = 1;
% sparse_target = 0.02;

useSingle = true;
prec = 'double';
if useSingle
    prec = 'single';
    train_data = single(train_data);
    lrate = single(lrate);
    mom = single(mom);
end
bv = zeros(data_dim,1, prec);
bh = randn(hid_num,1, prec) * 0.01;
W = randn(hid_num,data_dim, prec) * 0.01;

dbv = zeros(size(bv), prec);
dW = zeros(size(W), prec);
dbh = zeros(size(bh), prec);
vW = zeros(size(W), prec);
vbv = zeros(size(bv), prec);
vbh = zeros(size(bh), prec);

for epoch = 1:max_epoch
	if epoch > 5
		mom = 0.9;
	end
	
	order = randperm(data_sz);
	hsum = zeros(size(bh));
    
    for n = (1:data_sz / batchsz)
		% set visible from the train data
		chosen = order(1+batchsz*(n-1):batchsz*n);
		v = train_data(chosen,:)';

        h = bsxfun(@plus,  W*v, bh);
        h = 1 ./ (1 + exp(-h));  % sigmoid
        hsum = hsum + sum(h,2);
        h = cast(h > rand(size(h), prec), prec);  % sampling to binary
        
		% positive weight update
		dW = dW + h*v';
		dbv = dbv + sum(v,2);
		dbh = dbh + sum(h,2);
		
        for j = (1:CDk)
            v = bsxfun(@plus, W'*h, bv);
%             v = v + randn(size(v), prec);
            
            h = bsxfun(@plus,  W*v, bh);
            h = 1 ./ (1 + exp(-h));  % sigmoid
%             h = cast(h > rand(size(h), prec), prec);  % sampling to binary
        end
            
        % negative weight update
        dW = dW - h*v';
        dbv = dbv - sum(v,2);
        dbh = dbh - sum(h,2);
        
        vW = mom * vW + lrate * (dW / batchsz);
		vbv = mom * vbv + lrate * (dbv / batchsz);
		vbh = mom * vbh + lrate * (dbh / batchsz);
		
		W = W + vW;
		bv = bv + vbv;
		bh = bh + vbh;

		dW = zeros(size(W), prec);
		dbv = zeros(size(bv), prec);
		dbh = zeros(size(bh), prec);
	end
	fprintf(1,'epoch %d\n', epoch);
	
	if mod(epoch,1) == 0
		% plot learning process in real-time
        subplot(2,2,1);view_data(W,6,6,10,10);
		subplot(2,2,2);imagesc(h);
		subplot(2,2,3);hist(W(:),100);
		subplot(2,2,4);plot(hsum/data_sz);
		pause(0.1);
    end
end
