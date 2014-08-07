function pool = pool_greedy_multi(hdataA,hdataB, N, maxIter)
M = size(hdataA,2);
pool = zeros(M,N);
r = randperm(M);
for i = 1:N
    pool(r(i),i) = 1;
end
for i = 1:maxIter
    for k = 1:M
        m = randi(M);
        % create pool candidates
        poolTmp = pool;
        poolTmp(m,:) = 0;
        cost1 = cost_pool_variance(poolTmp,0.5,0,0,hdataA,hdataB);
        poolTmp(m,:) = 1;
        cost2 = cost_pool_variance(poolTmp,0.5,0,0,hdataA,hdataB);
        
        [x,y] = min(cost2 - cost1);
        poolTmp(m,:) = 0;
        poolTmp(m,y) = 1;
        pool = poolTmp;
        display_network(pool);
        drawnow
    end
end
