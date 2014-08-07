function pool = pool_by_centers(corr_to_centers)
data_dim = size(corr_to_centers,1);
num_centers = size(corr_to_centers,2);
pool = zeros(size(corr_to_centers));
for i = 1:data_dim
    [x,j] = max(corr_to_centers(i,:)); % find closest center
    pool(i,j) = 1;
end
% pool = resize_pool_fast(pool);