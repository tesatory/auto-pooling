prepare_cifar10

trainX = bsxfun(@rdivide, bsxfun(@minus, trainX, mean(trainX,2)), sqrt(var(trainX,[],2)+10));
testX = bsxfun(@rdivide, bsxfun(@minus, testX, mean(testX,2)), sqrt(var(testX,[],2)+10));

C = cov(trainX);
M = mean(trainX);
[V,D] = eig(C);
P = V * diag(sqrt(1./(diag(D) + 0.1))) * V';
trainX = bsxfun(@minus, trainX, M) * P;
testX = bsxfun(@minus, testX, M) * P;