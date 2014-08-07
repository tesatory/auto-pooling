function task_121120()
tic;
addpath ng_sparse_AE\
load('./work/121113_sparseAE_100hu_6x6color_cifar.mat');
prepare_cifar10
prepare_cifar10_motion
hdataA = extract_features_ae_nopool(dataA,W1,b1,rfSize,[16 16 3],M,P,1);
hdataB = extract_features_ae_nopool(dataB,W1,b1,rfSize,[16 16 3],M,P,1);

[pool, cost] = train_pool_ae(hdataA,hdataB,400,50);
save('./work/121120_pool_whole', 'pool','cost');

[pool, cost] = train_pool_ae(hdataA,hdataB,400,50,pool);
save('./work/121120_pool_whole2', 'pool','cost');

[pool, cost] = train_pool_ae(hdataA,hdataB,400,100,pool);
save('./work/121120_pool_whole3', 'pool','cost');

toc;
end