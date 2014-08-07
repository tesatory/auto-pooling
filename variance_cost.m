function cost = variance_cost(pool, hdataA, hdataB, sumPool, beta)
r = randperm(size(hdataA,1));

if sumPool
    pvalA = hdataA * pool;
    pvalB = hdataB * pool;
else
    pool = pool > 0;
    pvalA = zeros(size(hdataA,1),size(pool,2));
    pvalB = zeros(size(hdataB,1),size(pool,2));
    for pi = 1:size(pool,2)
        hdataA_sub = hdataA(:,pool(:,pi));
        hdataB_sub = hdataB(:,pool(:,pi));
        pvalA(:,pi) = max(hdataA_sub,[],2);
        pvalB(:,pi) = max(hdataB_sub,[],2);
    end   
end
% cost = mean(abs(pvalA - pvalB).^beta,1);
cost = mean(abs(pvalA - pvalB).^beta,1) ./ ...
    mean(abs(pvalA - pvalB(r,:)).^beta,1);
