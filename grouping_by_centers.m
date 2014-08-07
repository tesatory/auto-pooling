function [x]  = grouping_by_centers(W,N,max_iter)
data_dim = size(W,1);
x = zeros(data_dim,1);
r = randperm(data_dim);
x(r(1:N),1) = 1;

for t = 1:max_iter
    y = W * x;
    [a,i] = max(y.*x); % remove the center with max prob
    [a,j] = min(y + x*100); % add a center where prob is min
    x(i,1) = 0;
    x(j,1) = 1;
    display_network([x y]);
end

