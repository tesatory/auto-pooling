function WP = display_pool(pool, W1, maxpoolsz, thres)
WP = zeros(size(W1,2),1000,size(pool,2));
for i = 1:size(pool,2)
    A = W1(pool(:,i)>thres,:);
    WP(:,1:size(A,1),i) = A';
end
WP = WP(:,1:maxpoolsz,:);
WP = reshape(WP,size(W1,2),maxpoolsz*size(pool,2));
display_network(WP);