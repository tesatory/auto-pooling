function pool = pool_greedy_multi2(hdataA,hdataB, N,K, beta)
M = size(hdataA,2);
pool = zeros(M,K);
for k = 1:K
    pool(randi(M),k) = 1;
end
for n = 1:N
    for k = 1:K
        % create pool candidates
        poolTmp = repmat(pool(:,k),1,M) + eye(M);
        cost = variance_cost(poolTmp,hdataA,hdataB,true,0.5);
        if K > 1
%         cost2 = (sum(pool,2))';
        cost2 = (sum(pool,2) - pool(:,k))';
        cost = cost + beta * cost2 / sum(cost2);
        end
        [x,y] = min(cost);
        pool(:,k) = poolTmp(:,y);
    end
%     view_data_gray(pool', sqrt(M), sqrt(M), K, 1);
    display_network(pool);
    drawnow
end
