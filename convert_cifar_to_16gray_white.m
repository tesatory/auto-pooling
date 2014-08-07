trainX16g = zeros(50000,256);
for i = 1:50000
    a = reshape(trainX(i,:),32,32,3);
    a = rgb2gray(uint8(a));
    a = imresize(a,0.5,'bilinear');
    trainX16g(i,:) = double(a(:)')/256;
end
trainX16gw = bsxfun(@minus, trainX16g, M) * P;
testX16g = zeros(10000,256);
for i = 1:10000
    a = reshape(testX(i,:),32,32,3);
    a = rgb2gray(uint8(a));
    a = imresize(a,0.5,'bilinear');
    testX16g(i,:) = double(a(:)')/256;
end
testX16gw = bsxfun(@minus, testX16g, M) * P;
trainXC = 1./(1+exp(-bsxfun(@plus,W1*trainX16gw',b1)))';
testXC = 1./(1+exp(-bsxfun(@plus,W1*testX16gw',b1)))';