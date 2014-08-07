
task = zeros(10,3);
p = [1600 3200];
for n = 1:length(p)
    try
        EC_sub = EC(:,r(1:p(n)));
        pool = (EC_sub>0.1);
        pool = resize_pool_fast(pool);
        test_ae_pool;
        task(n,1) = p(n);
        task(n,2) = train_acc;
        task(n,3) = test_acc;
    catch err
    end
end


task2 = zeros(10,3);
p = [0.06 0.08 0.12 0.14];
for n = 1:length(p)
    try
        EC_sub = EC(:,r(1:800));
        pool = (EC_sub>p(n));
        pool = resize_pool_fast(pool);
        test_ae_pool;
        task2(n,1) = p(n);
        task2(n,2) = train_acc;
        task2(n,3) = test_acc;
    catch err
    end
end
