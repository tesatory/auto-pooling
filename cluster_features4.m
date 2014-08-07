function T = cluster_features4(centroids, dataAsum, dataBsum, cnum)
N = size(dataAsum,2);
data = dataAsum - dataBsum;

T = (1:N)';
for k = 1:(N - cnum)
    cost = sum(abs(data),1);
    cost = - bsxfun(@plus,cost',cost);
    
    for i = 1:N
        for j = 1:N
            cost(i,j) = cost(i,j) + sum(abs(data(:,i) + data(:,j)));
        end
    end
  
    [x,i] = min(cost);
    [x,j] = min(x);
    i = i(j);
keyboard
    data(:,i) = data(:,i) + data(:,j);
    data(:,j) = 0;
    
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