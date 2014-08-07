function task_121128()
load('./work/121128_video_patches_16x16gray_50k.mat');
load('./work/121122_sparse_1000hu_c5t1.mat');
C = cov(dataA);
M = mean(dataA);
[V,D] = eig(C);
P = V * diag(sqrt(1./(diag(D) + 0.001))) * V';
dataA = bsxfun(@minus,dataA,M) * P;
dataB = bsxfun(@minus,dataB,M) * P;
hdataA = 1 ./ (1 + exp(- bsxfun(@plus,W1 * dataA',b1)))';
hdataB = 1 ./ (1 + exp(- bsxfun(@plus,W1 * dataB',b1)))';
[pool, cost] = train_pool_ae(hdataA,hdataB,400,16,200);
save('./work/121128_task', 'pool','cost');
end