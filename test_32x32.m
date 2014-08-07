v = patches';
dist_sq = bsxfun(@plus, bsxfun(@plus, -2*W*v, sum(v.^2,1)), sum(W.^2,2));

% [x,y] = sort(dist_sq);
% h = zeros(hid_num, size(v,2));
% for i = 1:size(v,2)
%     h(y(1:N_active_hidden,i),i) = 1;
% end

dist_sq = dist_sq .* (dist_sq > 0);
dist = sqrt(dist_sq);
dist_avg = mean(dist,1);
h = max(0, bsxfun(@plus, - dist, dist_avg));

trainXC = h';

% standardize data
trainXC_mean = mean(trainXC);
trainXC_sd = sqrt(var(trainXC)+0.01);
trainXCs = bsxfun(@rdivide, bsxfun(@minus, trainXC, trainXC_mean), trainXC_sd);
trainXCs = [trainXCs, ones(size(trainXCs,1),1)];

% train classifier using SVM
C = 100;
addpath minFunc;
theta = train_svm_gd(trainXCs, trainY, C, 400, 0.001);

[val,labels] = max(trainXCs*theta, [], 2);
train_accuracy = 100 * (1 - sum(labels ~= trainY) / length(trainY));
fprintf('Train accuracy %f%%\n', train_accuracy);

%%%%% TESTING %%%%%

% compute testing features and standardize
v = patches_test';
dist_sq = bsxfun(@plus, bsxfun(@plus, -2*W*v, sum(v.^2,1)), sum(W.^2,2));

% [x,y] = sort(dist_sq);
% h = zeros(hid_num, size(v,2));
% for i = 1:size(v,2)
%     h(y(1:N_active_hidden,i),i) = 1;
% end

dist_sq = dist_sq .* (dist_sq > 0);
dist = sqrt(dist_sq);
dist_avg = mean(dist,1);
h = max(0, bsxfun(@plus, - dist, dist_avg));
testXC = h';

testXCs = bsxfun(@rdivide, bsxfun(@minus, testXC, trainXC_mean), trainXC_sd);
testXCs = [testXCs, ones(size(testXCs,1),1)];

% test and print result
[val,labels] = max(testXCs*theta, [], 2);
test_accuracy = 100 * (1 - sum(labels ~= testY) / length(testY));
fprintf('Test accuracy %f%%\n', test_accuracy);

