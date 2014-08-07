trainXC = 1 ./ (1 + exp(- bsxfun(@plus,ae.W1 * trainX',ae.b1)))';
trainXC = trainXC * W;
testXC = 1 ./ (1 + exp(- bsxfun(@plus,ae.W1 * testX',ae.b1)))';
testXC = testXC * W;
test
