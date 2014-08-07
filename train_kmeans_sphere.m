function [W] = train_kmeans_sphere(train_data,hid_num,max_epoch,M)
% meta parameters
data_dim = size(train_data,2);
data_sz = size(train_data,1);

useSingle = true;
prec = 'double';
if useSingle
    prec = 'single';
    train_data = single(train_data);
end
W = randn(hid_num,data_dim, prec);
W = bsxfun(@rdivide, W, sqrt(sum(W.^2,2)));
% W = train_data(randi([1 data_sz],hid_num,1),:);

v = train_data';
cost2 = inf;
for epoch = 1:max_epoch
    dist = - W * v;
    [x,y] = sort(dist);
    h = zeros(hid_num, size(v,2));
    for i = 1:size(v,2)
        h(y(1:M,i),i) = 1;
    end

    cost1 = - sum(sum(W .* (h * v'))) / data_sz;
    assert(cost2 > cost1);

    Wtmp = h * v';
    W = Wtmp + (Wtmp == 0) .* W;
    W = bsxfun(@rdivide, W, sqrt(sum(W.^2,2)));

    cost2 = - sum(sum(W .* (h * v'))) / data_sz;
    assert(cost2 < cost1);
    
    fprintf(1,'epoch %d cost %f\n', epoch, cost2);
    display_network(W');
    pause(0.1);
end
