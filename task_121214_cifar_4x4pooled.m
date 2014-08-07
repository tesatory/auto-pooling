function task_121214_cifar_4x4pooled()
prepare_cifar10
load('./work/121113_sparseAE_100hu_6x6color_cifar.mat')
tic
trainXC = extract_features_ae4x4(trainX,W1,b1,6,[32 32 3],M,P,1);
testXC = extract_features_ae4x4(testX,W1,b1,6,[32 32 3],M,P,1);
save('./work/121214_cifar_4x4_pooled','trainXC','testXC')
toc
end