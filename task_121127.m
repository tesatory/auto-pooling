% [dataA, dataB] = prepare_video_patches_color(16, 8, 30000,1);
load work/121113_sparseAE_100hu_6x6color_cifar.mat
hdataA = extract_features_ae_nopool(dataA,W1,b1,rfSize,[16 16 3],M,P,1);
hdataB = extract_features_ae_nopool(dataB,W1,b1,rfSize,[16 16 3],M,P,1);
[pool, cost] = train_pool_ae(hdataA,hdataB,400,4,10);
for i = 1:100
    [pool, cost] = train_pool_ae(hdataA,hdataB,400,4,10,pool);
end
