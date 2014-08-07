function [W,bv,bh] = train_rbRBM_nobias(train_data,hid_num,max_epoch,alpha)
% meta parameters
lrate = 0.01; % for W and bv
lrate2 = 0.01; % for bh

batchsz = 100;
CDk = 1;  % the number of gibbs sampling in contrastive divergence

data_dim = size(train_data,2);
data_sz = size(train_data,1);

h0 = 0.1;
mom = 0;

useSingle = true;
prec = 'double';
if useSingle
    prec = 'single';
    train_data = single(train_data);
    lrate = single(lrate);
    lrate2 = single(lrate2);
    mom = single(mom);
    h0 = single(h0);
end
bv = zeros(data_dim,1, prec);
bh = (randn(hid_num,1, prec) * 0.0 + 10);
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

        dist_sq = bsxfun(@plus, bsxfun(@plus, -2*W*v, sum(v.^2,1)), sum(W.^2,2));
        dist_sq_mean = mean(dist_sq,1);
        h = bsxfun(@plus, - dist_sq, dist_sq_mean * alpha)/2;
        h = 1 ./ (1 + exp(-h));  % sigmoid
        hsum = hsum + sum(h,2);
        h = cast(h > rand(size(h), prec), prec);  % sampling to binary
        hh = h;

        h = bsxfun(@minus, h, mean(h,1) * alpha);
		% positive weight update
		dW = dW + (h*v' - bsxfun(@times,W,sum(h,2)));
		dbv = dbv + h0 * sum(v,2);
		
        for j = (1:CDk)
            v = bsxfun(@plus, W'*h, bv * h0);
            v = bsxfun(@rdivide, v, sum(h,1) + h0);
%             v = v + bsxfun(@rdivide, randn(size(v), prec), sqrt(sum(h,1) + h0));
            
            dist_sq = bsxfun(@plus, bsxfun(@plus, -2*W*v, sum(v.^2,1)), sum(W.^2,2));
            dist_sq_mean = mean(dist_sq,1);
            h = bsxfun(@plus, - dist_sq, dist_sq_mean * alpha)/2;
            h = 1 ./ (1 + exp(-h));  % sigmoid
            h = cast(h > rand(size(h), prec), prec);  % sampling to binary

            h = bsxfun(@minus, h, mean(h,1) * alpha);
        end
            
        % negative weight update
        dW = dW - (h*v' - bsxfun(@times,W,sum(h,2)));
        dbv = dbv - h0 * sum(v,2);
        
        vW = mom * vW + lrate * (dW / batchsz);
		vbv = mom * vbv + lrate * (dbv / batchsz);
		
		W = W + vW;
		bv = bv + vbv;
		
		dW = zeros(size(W), prec);
		dbv = zeros(size(bv), prec);
	end
	fprintf(1,'epoch %d\n', epoch);
	
	if mod(epoch,1) == 0
		% plot learning process in real-time
        subplot(2,2,1);view_data_gray(W,10,10,10,10);
		subplot(2,2,2);imagesc(hh);
		subplot(2,2,3);hist(W(:),100);
		subplot(2,2,4);plot(hsum/data_sz);
		pause(0.1);
    end
end
