function T = cluster_features2(centroids, datasum, cnum,sz_cost)
N = size(datasum,2);
datasum = bsxfun(@minus,datasum,mean(datasum,1));
corr = datasum' * datasum / size(datasum,1);
data_std = std(datasum) + 0.0001;
corr = corr ./ (data_std' * data_std);
cost = 1 - corr;

for k = 1:N
    cost(k,k) = inf;
end

T = (1:N)';
for k = 1:(N - cnum)
    a = zeros(1,N);
    for m = 1:N
        a(1,m) = sum(T == T(m,1));
    end
    cost2 = bsxfun(@plus, a', a) * sz_cost;
    
    [x,i] = min(cost + cost2);
    [x,j] = min(x);
    i = i(j);

    cost(T == T(i,1),T == T(j,1)) = inf;
    cost(T == T(j,1),T == T(i,1)) = inf;
    T(T == T(j,1),1) = T(i,1);
end

j = 1;
TT = zeros(size(T));
for i = 1:N
    if sum(T == i) > 0
        TT(T == i,1) = j;
        j = j + 1;
    end
end
T = TT;

c = zeros(N * 10, size(centroids,2));
for i = 1:N
    v = centroids((T==i),:);
    v =  v(1:min(10,size(v,1)),:);
    c((1:size(v,1)) + (i-1)*10,:) = v;
end

view_data(c,6,6,40,ceil(10*max(T)/40));