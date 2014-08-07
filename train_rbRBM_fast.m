function [W,bv,bh] = train_rbRBM_fast(train_data,hid_num,max_epoch)
% meta parameters
lrate = 0.01; % for W and bv
lrate2 = 0.01; % for bh

batchsz = 100;
CDk = 1;  % the number of gibbs sampling in contrastive divergence

data_dim = size(train_data,2);
data_sz = size(train_data,1);

h0 = 0.1;
mom = 0;
temp = 1;

kmeans = false;
sample_1ofK = 3;
if kmeans || (sample_1ofK > 0)
    h0 = 0;
end

% sparse_cost = 1;
% sparse_target = 0.02;

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

        if kmeans
            dist = bsxfun(@plus, bsxfun(@plus, -2*W*v, sum(v.^2,1)), sum(W.^2,2));
            [x,y] = min(dist);
            h = zeros(hid_num, size(v,2));
            for i = 1:size(v,2)
                h(y(i),i) = 1;
            end
        else
            h = W*v;
            h = bsxfun(@plus, h, bh - sum(W.^2,2)/2);
            h = bsxfun(@plus, h, -sum(v.^2,1)/2);
            if sample_1ofK == 0
                h = 1 ./ (1 + exp(-h/temp));  % sigmoid
                hsum = hsum + sum(h,2);                
                %             if sparse_cost > 0
                %                 % make sparse
                %                 dbh = dbh - sparse_cost * (mean(h,2) - sparse_target) ...
                %                    .* sum(h .* (1-h),2);
                %             end
                %             if sparse_cost2 > 0
                %                 a = bsxfun(@plus, h .* (1-h), mean(h,1) - sparse_target);
                %                 dbh = dbh - sparse_cost2 * sum(a,2);
                %                 dW = dW - sparse_cost2 * (a * v' - bsxfun(@times, W, sum(a,2)));
                %             end                
                h = cast(h > rand(size(h), prec), prec);  % sampling to binary
            else
                h = sample_MofK(h,sample_1ofK);
            end
        end
        
%         energy(n) = mean( ...
%             0.5 * sum(v.^2,1) .* (sum(h,1)+h0) ...
%             + 0.5 * sum(W.^2,2)' * h ...
%             - sum(h .* (W * v),1) ...
%             - bh' * h ...
%         );
        

		% positive weight update
		dW = dW + (h*v' - bsxfun(@times,W,sum(h,2)));
		dbv = dbv + h0 * sum(v,2);
		dbh = dbh + sum(h,2);
		
        if kmeans == false
            for j = (1:CDk)
                v = bsxfun(@plus, W'*h, bv * h0);
                v = bsxfun(@rdivide, v, sum(h,1) + h0);
                v = v + bsxfun(@rdivide, randn(size(v), prec), sqrt(sum(h,1) + h0) * temp);
                
                h = W*v;
                h = bsxfun(@plus, h, bh - sum(W.^2,2)/2);
                h = bsxfun(@plus, h, -sum(v.^2,1)/2);
                if sample_1ofK == 0                     
                    h = 1 ./ (1 + exp(-h/temp));  % sigmoid
                    h = cast(h > rand(size(h), prec), prec);  % sampling to binary
                else
                    h = sample_MofK(h,sample_1ofK);
                end
            end
            
            % negative weight update
            dW = dW - (h*v' - bsxfun(@times,W,sum(h,2)));
            dbv = dbv - h0 * sum(v,2);
            dbh = dbh - sum(h,2);
        end
        dbh = ones(size(dbh)) * mean(dbh);
		vW = mom * vW + lrate * (dW / batchsz);
		vbv = mom * vbv + lrate * (dbv / batchsz);
		vbh = mom * vbh + lrate2 * (dbh / batchsz);
		
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
        [x,y] = sort(bh);
%         if size(W,2) == 108
            subplot(2,2,1);view_data_gray(W(y,:),32,32,10,10);
%         end
		subplot(2,2,2);imagesc(h(y,:));
		subplot(2,2,3);plot(bh(y));
		subplot(2,2,4);plot(hsum(y)/data_sz);
% 		subplot(2,2,4);
%         scatter(tdpca(1:1000,1),tdpca(1:1000,2),'.b');
%         hold on
%         wp = W * Mpca;
%         scatter(wp(:,1),wp(:,2),'.r');
%         hold off;
		pause(0.1);
    end
    
%     if mod(epoch,10) == 0
%         Whist(:,:,epoch/10) = W;
%     end
end
