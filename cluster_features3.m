function T = cluster_features3(centroids, datasum, cnum,csize)
N = size(datasum,2);
datasum = bsxfun(@minus,datasum,mean(datasum,1));
corr = datasum' * datasum / size(datasum,1);
data_std = std(datasum) + 0.0001;
corr = corr ./ (data_std' * data_std);
cost = 1 - corr;

for k = 1:N
    cost(k,k) = inf;
end

T = zeros(N, cnum);

for k = 1:cnum
    m = randi(N);
    T(m,k) = 1;
    [x,i] = sort(cost(m,:));
    T(i(1:csize-1),k) = 1;
end

c = zeros(cnum * 10, size(centroids,2));
for i = 1:cnum
    v = centroids((T(:,i)==1),:);
    v =  v(1:min(10,size(v,1)),:);
    c((1:size(v,1)) + (i-1)*10,:) = v;
end

view_data(c,6,6,40,ceil(10*cnum/40));