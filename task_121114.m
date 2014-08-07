try
    EC_sub = EC(:,r(1:400));
    pool = EC_sub;
    pool = resize_pool_fast(pool);
    test_ae_pool;
    train_acc_pos = train_acc;
    test_acc_pos = test_acc;
catch err
end

try
    EC_sub = EC(:,r(1:800));
    pool = EC_sub;
    pool = resize_pool_fast(pool);
    test_ae_pool;
    train_acc_pos2 = train_acc;
    test_acc_pos2 = test_acc;
catch err
end

try
    EC_sub = EC(:,r(1:400));
    pool = (EC_sub>0.1);
    pool = resize_pool_fast(pool);
    test_ae_pool;
    train_acc_thres = train_acc;
    test_acc_thres = test_acc;
catch err
end

try
    EC_sub = EC(:,r(1:800));
    pool = (EC_sub>0.1);
    pool = resize_pool_fast(pool);
    test_ae_pool;
    train_acc_thres2 = train_acc;
    test_acc_thres2 = test_acc;
catch err
end

try
    EC_sub = EC(:,r(1:400));
    pool = (EC_sub).^2;
    pool = resize_pool_fast(pool);
    test_ae_pool;
    train_acc_sq = train_acc;
    test_acc_sq = test_acc;
catch err
end

try
    EC_sub = EC(:,r(1:800));
    pool = (EC_sub).^2;
    pool = resize_pool_fast(pool);
    test_ae_pool;
    train_acc_sq2 = train_acc;
    test_acc_sq2 = test_acc;
catch err
end


task = zeros(10,3);
p = [1600 3200 6400];
for n = 1:length(p)
    try
        EC_sub = EC(:,r(1:p(n)));
        pool = pool_by_centers(EC_sub);
        test_ae_pool;
        task(n,1) = p(n);
        task(n,2) = train_acc;
        task(n,3) = test_acc;
    catch err
    end
end
