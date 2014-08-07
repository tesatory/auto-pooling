function pool = pool_greedy(hdataA,hdataB, N)
M = size(hdataA,2);
pool = zeros(M,1);
pool(randi(M),1) = 1;
for n = 1:N
    % create pool candidates
    poolTmp = repmat(pool,1,M) + eye(M);
%     cost = variance_cost(poolTmp,hdataA,hdataB,true,0.5);
    cost = cost_pool_variance(poolTmp,0.5,0,0,hdataA,hdataB);
    [x,y] = sort(cost);
    pool(y(1:3),1) = pool(y(1:3),1) + 1;
    subplot(1,2,1);imagesc(reshape(pool,sqrt(M),sqrt(M)));
    subplot(1,2,2);imagesc(reshape(-cost,sqrt(M),sqrt(M)));
    drawnow
end
