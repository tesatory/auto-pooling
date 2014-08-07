prepare_cifar10_patches

cost_list = [1 3 10 30 100];
for cost = 1:5
    for target = 1:5
        sparse_cost = cost_list(cost);
        sparse_target = 0.02 * target;
        [W,bv,bh] = train_rbRBM_fast(patches,100,100,sparse_target,sparse_cost);
        Wall(:,:,cost,target) = W;
        bhall(:,cost,target) = bh;
        bvall(:,cost,target) = bv;
    end
end