function task_121211_hdata_batches()
load('./work/121211_vimio_full_100k_single.mat')
load('./work/121113_sparseAE_100hu_6x6color_cifar.mat')
r = randperm(100000);
dataA = dataA(r,:) * 256;
dataB = dataB(r,:) * 256;
tic
for k = 1:100
    fprintf('processing %d\n',k)
    hdataA = extract_features_ae_nopool(dataA(1+(k-1)*1000:k*1000,:),W1,b1,6,[32 32 3],M,P,1)';
    hdataB = extract_features_ae_nopool(dataB(1+(k-1)*1000:k*1000,:),W1,b1,6,[32 32 3],M,P,1)';
    save(['/mnt/ssd0/sainaa/vimio_hdataAB_27x27x100_' int2str(k)],'hdataA','hdataB')
    toc
end
end