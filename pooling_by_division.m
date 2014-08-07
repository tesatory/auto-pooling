function pool = pooling_by_division(hdataA, hdataB, centroids, sz, rfSize, clusters)
psz = sz - rfSize + 1;
K = max(clusters); % number of clusters
if max(clusters) == 1
    K = size(clusters,2);
end
pool = zeros(psz^2, size(centroids,1), 4, K);
N = size(hdataA,1); % number of data
hdataA = reshape(hdataA, N, psz^2, size(centroids,1));
hdataB = reshape(hdataB, N, psz^2, size(centroids,1));

for k = 1:K
    k
    if max(clusters) == 1
        c = (clusters(:,k) == 1);
    else
        c = (clusters == k);
    end
    hdataA1 = hdataA(:,:,c);
    hdataB1 = hdataB(:,:,c);
    M = sum(c); % cluster size
    hdataA2 = reshape(hdataA1, N, psz^2 * M);
    hdataB2 = reshape(hdataB1, N, psz^2 * M);
    [WA,WB] = divide_region_by_line(ones(1,psz^2),psz);
    WA2 = repmat(WA,1,M);
    WB2 = repmat(WB,1,M);
    sA=calc_distance(WA2,hdataA2,hdataB2);
    sB=calc_distance(WB2,hdataA2,hdataB2);
    [x,i]=min(sA+sB);
    [WAA,WAB] = divide_region_by_line(WA(i,:),psz);
    [WBA,WBB] = divide_region_by_line(WB(i,:),psz);
    WAA2 = repmat(WAA,1,M);
    WAB2 = repmat(WAB,1,M);
    WBA2 = repmat(WBA,1,M);
    WBB2 = repmat(WBB,1,M);
    sAA=calc_distance(WAA2,hdataA2,hdataB2);
    sAB=calc_distance(WAB2,hdataA2,hdataB2);
    sBA=calc_distance(WBA2,hdataA2,hdataB2);
    sBB=calc_distance(WBB2,hdataA2,hdataB2);
    [x,ia]=min(sAA+sAB);
    [x,ib]=min(sBA+sBB);
    pool(:,c,1,k) = repmat(WAA(ia,:)', 1, M);
    pool(:,c,2,k) = repmat(WAB(ia,:)', 1, M);
    pool(:,c,3,k) = repmat(WBA(ib,:)', 1, M);
    pool(:,c,4,k) = repmat(WBB(ib,:)', 1, M);
end
pool = reshape(pool, psz^2 * size(centroids,1), 4 * K);