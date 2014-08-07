function task_121129_conv_pool_train()
load('./work/121129_video_patches_16x16gray_50k_nostill.mat');
load('./work/121128_sparse_cifar_7x7gray_t2c5.mat');
load('./work/121129_task_conv_pool_train.mat');
hdataA = extract_features_ae_nopool(dataA,W1,b1,7,[16 16 1],M,P,1);
hdataB = extract_features_ae_nopool(dataB,W1,b1,7,[16 16 1],M,P,1);
[pool, cost] = train_pool_ae(hdataA,hdataB,400,4,500,pool);
save('./work/121130_task_conv_pool_train', 'pool','cost');
end
