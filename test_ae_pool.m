trainXC = extract_features_ae_pooled(trainX,W1,b1,pool,rfSize,[32 32 3],M,P,stride);
testXC = extract_features_ae_pooled(testX,W1,b1,pool,rfSize,[32 32 3],M,P,stride);

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
train_acc = 100 * (1 - sum(labels ~= trainY) / length(trainY));
fprintf('Train accuracy %f%%\n', train_acc);

%%%%% TESTING %%%%%
testXCs = bsxfun(@rdivide, bsxfun(@minus, testXC, trainXC_mean), trainXC_sd);
testXCs = [testXCs, ones(size(testXCs,1),1)];

% test and print result
[val,labels] = max(testXCs*theta, [], 2);
test_acc = 100 * (1 - sum(labels ~= testY) / length(testY));
fprintf('Test accuracy %f%%\n', test_acc);

