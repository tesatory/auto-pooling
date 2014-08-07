function T = cluster_features(centroids, data1, data2, cnum)
hdataAsum = zeros(size(data1,1), size(centroids,1));
M = size(data1,2) / size(centroids,1);
for k=1:size(centroids,1)
    hdataAsum(:,k) = sum(data1(:,(1:M)+(k-1)*M),2);
end
hdataAsumM = bsxfun(@minus,hdataAsum,mean(hdataAsum,1));

Y = pdist(hdataAsumM','correlation');
Z = linkage(Y,'single');
T = cluster(Z,'maxclust',cnum);
c = zeros(cnum * 10, size(centroids,2));
for i = 1:cnum
    v = centroids((T==i),:);
    v =  v(1:min(10,size(v,1)),:);
    c((1:size(v,1)) + (i-1)*10,:) = v;
end
view_data(c,6,6,40,25);