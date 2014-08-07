prepare_cifar10

trainX2 = zeros(50000, 16*16);
for n = 1:50000
    a = reshape(trainX(n,:),32,32,3);
    a = rgb2gray(a/256)*256;
    a = imresize(a,0.5);
    trainX2(n,:) = a(:)';
end

testX2 = zeros(10000, 16*16);
for n = 1:10000
    a = reshape(testX(n,:),32,32,3);
    a = rgb2gray(a/256)*256;
    a = imresize(a,0.5);
    testX2(n,:) = a(:)';
end

trainX = trainX2;
testX = testX2;
clear trainX2 testX2