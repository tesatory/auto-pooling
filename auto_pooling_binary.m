function W = auto_pooling_binary(data1, data2)
data = data1 - data2;
% data = bsxfun(@rdivide,data,sqrt(sum(data.^2,2))); % normalize

M = size(data,2);
K = sqrt(M);
W = eye(M);

N = size(W,1);
cost = mean(bsxfun(@rdivide, abs(W * data'), sqrt(sum(W.^2,2))), 2);
% cost = mean(abs(W * data') ./ (sqrt(sum(W.^2,2)) * sqrt(sum(data.^2,2))'), 2);
cost_change = zeros(N,N) + inf;
for i = 1:N-1
    for j = i+1:N
        cost_ij = mean(abs((W(i,:)+W(j,:)) * data')) / sqrt(sum((W(i,:)+W(j,:)).^2));
%         cost_ij = mean(abs((W(i,:)+W(j,:)) * data') ./ sqrt(sum(data.^2,2))') / sqrt(sum((W(i,:)+W(j,:)).^2));
        cost_change(i,j) = cost_ij - cost(i) - cost(j);
        cost_change(j,i) = cost_ij - cost(i) - cost(j);
    end
end

for iter = 1:117
    sum(cost)
%     [x,y] = sort(sum(W,2),'desc'); % sort by area
    view_data_gray(W,K,K,K,K);
    pause(0.1);
    
    [x,i] = min(cost_change);
    [x,j] = min(x);
    i = i(j);
    
    % make i = i + j
    W(i,:) = W(i,:) + W(j,:);
    cost(i,1) = cost_change(i,j) + cost(i) + cost(j);
    
    % make j = 0
    W(j,:) = 0;
    cost(j,1) = 0;
    
    % update cost
    cost_change(:,j) = 0;
    cost_change(j,:) = 0;
    for j = 1:N
        if sum(W(j,:)) > 0
            cost_ij = mean(abs((W(i,:)+W(j,:)) * data')) / sqrt(sum((W(i,:)+W(j,:)).^2));
%             cost_ij = mean(abs((W(i,:)+W(j,:)) * data') ./ sqrt(sum(data.^2,2))') / sqrt(sum((W(i,:)+W(j,:)).^2));
            cost_change(i,j) = cost_ij - cost(i) - cost(j);
            cost_change(j,i) = cost_ij - cost(i) - cost(j);
        end
    end
    cost_change(i,i) = inf;
end
