function [W,A] = train_multi_kmeans_grouped3(train_data, group_num, group_sz, max_epoch,A)
data_dim = size(train_data,2);
data_sz = size(train_data,1);

useSingle = true;
prec = 'double';
if useSingle
    prec = 'single';
    train_data = single(train_data);
end
W = randn(group_sz,data_dim,group_num, prec) * 0.1;
% W = train_data(randi([1 data_sz],hid_num,1),:);
% A = zeros(data_dim, group_num, prec);
% for i = 1:data_dim
%     A(i,randi(group_num)) = 1;
% end

cost = zeros(group_num,max_epoch);

v = train_data';
for epoch = 1:max_epoch
    Q = zeros(data_dim, group_num);
    for g = 1:group_num
        dist_sq = bsxfun(@plus, bsxfun(@plus, ...
            -2*W(:,:,g)*bsxfun(@times,v,A(:,g)), ...
            sum(bsxfun(@times,v.^2,A(:,g)),1)), ...
            W(:,:,g).^2 * A(:,g));
        [x,y] = sort(dist_sq);
        cost(g,epoch) = mean(x(1,:));
        h = zeros(group_sz, size(v,2));
        for i = 1:size(v,2)
            h(y(1,i),i) = 1;
        end
        
        % make weight center
        W(:,:,g) = bsxfun(@rdivide,h,sum(h,2)) * v';
        
        Q(:,g) = (v.^2) * sum(h,1)' + sum((W(:,:,g).^2)'*h,2) ...
            - 2*sum((W(:,:,g)'*h) .* v,2);
        
    end
    
    [x,y] = sort(Q,2);
    A = zeros(data_dim, group_num, prec);
    for i = 1:data_dim
        A(i,y(i)) = 1;
    end
    
    
    fprintf(1,'epoch %d cost %f \n', epoch, sum(cost(:,epoch)));
    
    if mod(epoch,1) == 0
        % plot learning process in real-time
        for g = 1:group_num
            subplot(2,group_num,g);display_network(W(:,:,g)');title('W');
        end
        subplot(2,group_num,1+group_num);display_network(A);title('A');
        subplot(2,group_num,2+group_num);plot(cost');legend('show');
        pause(0.1);
    end
end