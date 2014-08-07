function [W1,W2,b1,b2] = train_sparse_autoencoder_binary(train_data,hiddenSize,max_epoch)
visibleSize = size(train_data,2);
data_sz = size(train_data,1);

% meta parameters
lrate = 1;
batchsz = 10000;
sparsityParam = 0.01;   % desired average activation of the hidden units.
                     % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		     %  in the lecture notes). 
lambda = 0.0001;     % weight decay parameter       
beta = 3;            % weight of sparsity penalty term    
moment = 0;
% train a Denoising AutoEncoder

r  = sqrt(6) / sqrt(hiddenSize+visibleSize+1);   % we'll choose weights uniformly from the interval [-r, r]
W1 = rand(hiddenSize, visibleSize) * 2 * r - r;
W2 = rand(visibleSize, hiddenSize) * 2 * r - r;

b1 = zeros(hiddenSize, 1);
b2 = zeros(visibleSize, 1);

W1speed = zeros(size(W1));
W2speed = zeros(size(W2));
b1speed = zeros(size(b1));
b2speed = zeros(size(b2));
cost = 0;
for epoch = 1:max_epoch    
    order = randperm(data_sz);  % shuffle training data
    if epoch == 5
        moment = 0.9;
    end
	for n = (1:data_sz / batchsz)
		% set visible from the train data
		chosen = order(1+batchsz*(n-1):batchsz*n);
		data = train_data(chosen,:)';

        a1 = bsxfun(@plus, W1 * data, b1);
        h1 = 1./(1 + exp(-a1));
        a2 = bsxfun(@plus, W2 * h1, b2);
        h2 = 1./(1 + exp(-a2));
        
        cost = mean(sum(0.5*(h2 - data).^2,1),2);
        
        cost_sparse = sparsityParam * log(sparsityParam ./ mean(h1,2)) + ...
            (1 - sparsityParam) * log((1-sparsityParam) ./ (1-mean(h1,2)));
        cost = cost + beta * sum(cost_sparse);
        
        t2 = (h2 - data)  .* h2 .* (1 - h2) / batchsz;
        b2grad = sum(t2,2);
        W2grad = t2 * h1';
        
        t1 = W2' * t2;
        t3 = beta * (- sparsityParam ./ mean(h1,2) + ...
            (1 - sparsityParam) ./ (1 - mean(h1,2))) / batchsz;
        t1 = bsxfun(@plus, t1, t3);
        t1 = t1 .* h1 .* (1 - h1);
        b1grad = sum(t1,2);
        W1grad = t1 * data';
        
        % add weight regularization cost
        cost = cost + lambda*0.5*(sum(sum(W1.^2)) + sum(sum(W2.^2)));
        W1grad = W1grad + lambda * W1;
        W2grad = W2grad + lambda * W2;
        
        W1speed = W1speed * moment + W1grad;
        W2speed = W2speed * moment + W2grad;
        b1speed = b1speed * moment + b1grad;
        b2speed = b2speed * moment + b2grad;
        
        W1 = W1 - lrate * W1speed;
        W2 = W2 - lrate * W2speed;
        b1 = b1 - lrate * b1speed;
        b2 = b2 - lrate * b2speed;        
    end
    fprintf(1,'epoch %d cost %f\r',epoch, cost);

    if mod(epoch,50) == 0
        subplot(2,3,6);hist(W1(:),40);
        subplot(2,3,1);display_network(data);
        subplot(2,3,2);imagesc(h1);
        subplot(2,3,3);display_network(h2);
        subplot(2,3,4);display_network(W1');
        subplot(2,3,5);display_network(W2);
        
        pause(0.1);
    end
end
