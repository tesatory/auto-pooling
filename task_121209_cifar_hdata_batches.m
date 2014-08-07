function task_121209_cifar_hdata_batches()
prepare_cifar10
load('./work/121113_sparseAE_100hu_6x6color_cifar.mat')
tic
for k = 1:50
    fprintf('processing train %d\n',k)
    trainXC = extract_features_ae_nopool(trainX(1+(k-1)*1000:k*1000,:),W1,b1,6,[32 32 3],M,P,1)';
    save(['./work/cifar_hdataAB_27x27x100_train_' int2str(k)],'trainXC')
    toc
end
for k = 1:10
    fprintf('processing test %d\n',k)
    testXC = extract_features_ae_nopool(testX(1+(k-1)*1000:k*1000,:),W1,b1,6,[32 32 3],M,P,1)';
    save(['./work/cifar_hdataAB_27x27x100_test_' int2str(k)],'testXC')
    toc
end
end