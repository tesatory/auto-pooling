function W = auto_pooling_binary2(data1, data2)
data = data1 - data2;
dataX = data1(randperm(size(data1,1)),:) - data2;
% data = bsxfun(@rdivide,data,sqrt(sum(data.^2,2))); % normalize

M = size(data,2);
N = M / 121;
M = 121;
K = sqrt(M);
W = zeros(1,121);
W(1,randi(121)) = 1;

for iter = 1:121
    cost = zeros(121,1) + inf;
    for i = 1:121
        if W(1,i) == 0
            WN = W;
            WN(1,i) = WN(1,i) + 1;
%             cost(i,1) = mean(abs(WN * data')) / sqrt(sum(WN.^2));
            cost(i,1) = mean(abs(WN * data')) - mean(abs(WN * dataX'));
%             cost(i,1) = mean(abs(WN * data')) / std(data1 * WN')';
%             cost(i,1) = -std(data1 * WN')';
        end
    end
    
    [x,i] = min(cost);
    W(1,i) = W(1,i) + 1;
    if N == 1
        view_data_gray(W,K,K,1,1);
    else
        view_data_gray(W,K,K,20,10);
    end
    pause(0.1);
end
