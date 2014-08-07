prepare_cifar10
trainX = bsxfun(@rdivide, bsxfun(@minus, trainX, mean(trainX,2)), sqrt(var(trainX,[],2)+10));

% whiten
C = cov(trainX);
M = mean(trainX);
[V,D] = eig(C);
P = V * diag(sqrt(1./(diag(D) + 0.1))) * V';
trainX = bsxfun(@minus, trainX, M) * P;

[W1,W2,b1,b2] = train_sparseAE_linear(trainX,1000);
save work\121122_sparse_cifar32_1000hu_c5t1 W1 W2 b1 b2