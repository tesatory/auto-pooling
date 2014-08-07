function [W,bv,bh,A] = train_rbRBM_slope(train_data,hid_num,max_epoch)
% meta parameters
lrate = 0.01; % for W and bv
lrate2 = 0.01; % for bh
lrate3 = 0.01;
drate = 0.001;

batchsz = 100;
CDk = 1;  % the number of gibbs sampling in contrastive divergence

data_dim = size(train_data,2);
data_sz = size(train_data,1);

mom = 0;
% sparse_cost = 1;
% sparse_target = 0.02;
h0 = 0.1;
temp = 1;

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
bh = ones(hid_num,1, prec) * 10;
W = randn(hid_num,data_dim, prec) * 0.01;
% rp = randperm(data_sz);
% W = train_data(rp(1:hid_num),:);
A = ones(hid_num,data_dim, prec);
AD = 0.5;
AS = sum(A.^AD,2);

dbv = zeros(size(bv), prec);
dW = zeros(size(W), prec);
dA = zeros(size(A), prec);
dbh = zeros(size(bh), prec);
vW = zeros(size(W), prec);
vA = zeros(size(A), prec);
vbv = zeros(size(bv), prec);
vbh = zeros(size(bh), prec);

% [Mpca,tdpca] = princomp(train_data);

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

        h = (A.*W) * v;
        h = bsxfun(@plus, h, bh - sum((W.^2) .* A,2)/2);
        h = bsxfun(@plus, h, - A * (v.^2)/2);
        h = 1 ./ (1 + exp(-h/temp));  % sigmoid
        hsum = hsum + sum(h,2);
        h = cast(h > rand(size(h), prec), prec);  % sampling to binary        

		% positive weight update
		dW = dW + (A.*(h*v') - bsxfun(@times,A.*W,sum(h,2)));
		dbv = dbv + h0 * sum(v,2);
		dbh = dbh + sum(h,2);
        dA = dA - (0.5*h*(v.^2)' + 0.5*bsxfun(@times, W.^2, sum(h,2)) - W.*(h*v'));
        
        for j = (1:CDk)
            v = bsxfun(@plus, (A.*W)'*h, bv * h0);
            v = bsxfun(@rdivide, v, A'*h + h0);
            v = v + bsxfun(@rdivide, randn(size(v), prec), sqrt((A'*h + h0) / temp));
            
            h = (A.*W) * v;
            h = bsxfun(@plus, h, bh - sum((W.^2) .* A,2)/2);
            h = bsxfun(@plus, h, - A * (v.^2)/2);
            h = 1 ./ (1 + exp(-h/temp));  % sigmoid
            h = cast(h > rand(size(h), prec), prec);  % sampling to binary
        end
        
        % negative weight update
        dW = dW - (A.*(h*v') - bsxfun(@times,A.*W,sum(h,2)));
        dbv = dbv - h0 * sum(v,2);
        dbh = dbh - sum(h,2);
        dA = dA + (0.5*h*(v.^2)' + 0.5*bsxfun(@times, W.^2, sum(h,2)) - W.*(h*v'));
        
%         dbh = ones(size(dbh)) * mean(dbh);
		vW = mom * vW + lrate * (dW / batchsz);
		vbv = mom * vbv + lrate * (dbv / batchsz);
		vbh = mom * vbh + lrate2 * (dbh / batchsz);
		
		W = W + vW - drate * W;
		bv = bv + vbv;
		bh = bh + vbh;

% 		vA = mom * vA + lrate3 * (dA / batchsz);
%         A = A + vA;
%         A = A .* (A > 0);
% 		A = bsxfun(@rdivide, A, (sum(A.^AD,2) ./ AS).^(1/AD));
        
		dW = zeros(size(W), prec);
		dbv = zeros(size(bv), prec);
		dbh = zeros(size(bh), prec);
        dA = zeros(size(A), prec);
    end
	fprintf(1,'epoch %d\n', epoch);
	
	if mod(epoch,1) == 0
		% plot learning process in real-time
        %[x,y] = sort(bh);
        y = 1:hid_num;
        subplot(2,3,1);display_network(W(y,:)');
		subplot(2,3,2);display_network(A(y,:)');
		subplot(2,3,3);display_network(A(y,:)'.*W(y,:)');
		subplot(2,3,4);plot(hsum(y)/data_sz);
		subplot(2,3,5);imagesc(h(y,:));
		pause(0.1);
    end

end
