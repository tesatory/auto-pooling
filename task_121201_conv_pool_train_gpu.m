function task_121201_conv_pool_train_gpu()
load('./work/121129_video_patches_16x16gray_50k_nostill.mat');
load('./work/121128_sparse_cifar_7x7gray_t2c5.mat');
hdataA = single(extract_features_ae_nopool(dataA,W1,b1,7,[16 16 1],M,P,1));
hdataB = single(extract_features_ae_nopool(dataB,W1,b1,7,[16 16 1],M,P,1));
pool = zeros(10,10,100,4,100);
for k = 1:100
    pool(1:5,1:5,k,1,k) = 1;
    pool(1:5,6:10,k,2,k) = 1;
    pool(6:10,1:5,k,3,k) = 1;
    pool(6:10,6:10,k,4,k) = 1;
end
pool = reshape(pool,10000,400);
pool = train_pool_ae2(hdataA,hdataB,400,4,1000,true,true,pool);
pool = gather(pool);
save('./work/121201_task_conv_pool_train_gpu1', 'pool');

pool = train_pool_ae2(hdataA,hdataB,400,4,1000,true,true);
pool = gather(pool);
save('./work/121201_task_conv_pool_train_gpu2', 'pool');
end