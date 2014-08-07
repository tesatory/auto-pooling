% extract training features
% trainXC = extract_pooled(trainX, centroids, pool, rfSize, CIFAR_DIM, M,P);

trainXC = zeros(size(trainX,1), size(pool,2));
for k = 1:bsz:size(trainX,1)
    fprintf('process: %d / %d\n', k, size(trainX,1));
    m = min(size(trainX,1),k+bsz-1);
    patches = get_patches(trainX(k:m,:), rfSize, CIFAR_DIM, M, P);
    patches = extract_features_from_patches(patches, centroids,rfSize, CIFAR_DIM);
    trainXC(k:m,:) = patches * pool;
end

% standardize data
trainXC_mean = mean(trainXC);
trainXC_sd = sqrt(var(trainXC)+0.01);
trainXCs = bsxfun(@rdivide, bsxfun(@minus, trainXC, trainXC_mean), trainXC_sd);
trainXCs = [trainXCs, ones(size(trainXCs,1),1)];

% train classifier using SVM
C = 100;
addpath minFunc;
theta = train_svm(trainXCs, trainY, C);

[val,labels] = max(trainXCs*theta, [], 2);
fprintf('Train accuracy %f%%\n', 100 * (1 - sum(labels ~= trainY) / length(trainY)));

%%%%% TESTING %%%%%

% compute testing features and standardize
% testXC = extract_pooled(testX, centroids, pool, rfSize, CIFAR_DIM, M,P);
testXC = zeros(size(testX,1), size(pool,2));
for k = 1:bsz:size(testX,1)
    fprintf('process: %d / %d\n', k, size(testX,1));
    m = min(size(testX,1),k+bsz-1);
    patches = get_patches(testX(k:m,:), rfSize, CIFAR_DIM, M, P);
    patches = extract_features_from_patches(patches, centroids,rfSize, CIFAR_DIM);
    testXC(k:m,:) = patches * pool;
end
testXCs = bsxfun(@rdivide, bsxfun(@minus, testXC, trainXC_mean), trainXC_sd);
testXCs = [testXCs, ones(size(testXCs,1),1)];

% test and print result
[val,labels] = max(testXCs*theta, [], 2);
fprintf('Test accuracy %f%%\n', 100 * (1 - sum(labels ~= testY) / length(testY)));

