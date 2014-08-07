function [W1,W2,b1,b2] = train_AE(train_data,hiddenSize,max_epoch)
visibleSize = size(train_data,2);
data_sz = size(train_data,1);

% meta parameters
lrate = 0.01;
batchsz = 100;
lambda = 0.000;     % weight decay parameter       
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

costs = zeros(max_epoch,1);

% order = randperm(data_sz);  % shuffle training data
% chosen = order(1:100);
% data = train_data(chosen,:)';
% noise = (rand(size(data)) > noise_rate);
% data_noisy = data .* noise;
% % noise = (1-noise) .* (rand(size(data)) > 0.5);
% % data_noisy = data_noisy + noise;
% 
% theta = [W1(:) ; W2(:) ; b1(:) ; b2(:)];
% [cost, grad] = cost_DA(theta, visibleSize, hiddenSize, ...
%     lambda, data, data_noisy);
% 
% numgrad = computeNumericalGradient( @(x) cost_DA(x, visibleSize, ...
%     hiddenSize, lambda, data, data_noisy), theta);
% diff = norm(numgrad-grad)/norm(numgrad+grad);
% display(diff);
% assert(diff < 1e-9);

for epoch = 1:max_epoch    
    order = randperm(data_sz);  % shuffle training data
    if epoch == 5
        moment = 0.9;
    end
	for n = (1:data_sz / batchsz)
		% set visible from the train data
		chosen = order(1+batchsz*(n-1):batchsz*n);
		data = train_data(chosen,:)';
        
        theta = [W1(:) ; W2(:) ; b1(:) ; b2(:)];
        [cost, grad] = cost_AE(theta, visibleSize, hiddenSize, ...
            lambda, data);
        costs(epoch) = cost;
        
        W1grad = reshape(grad(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
        W2grad = reshape(grad(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
        b1grad = grad(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
        b2grad = grad(2*hiddenSize*visibleSize+hiddenSize+1:end);

        W1speed = W1speed * moment - W1grad;
        W2speed = W2speed * moment - W2grad;
        b1speed = b1speed * moment - b1grad;
        b2speed = b2speed * moment - b2grad;
        
        W1 = W1 + lrate * W1speed;
        W2 = W2 + lrate * W2speed;
        b1 = b1 + lrate * b1speed;
        b2 = b2 + lrate * b2speed;    
        
        W1 = (W1 + W2') / 2;
        W2 = W1';
    end
    fprintf(1,'epoch %d cost %f\r',epoch, cost);

    if mod(epoch,1) == 0
%         subplot(2,3,6);hist(W1(:),40);
%         subplot(2,3,1);display_network(data);
%         subplot(2,3,2);display_network(data_noisy);
%         subplot(2,3,3);display_network(h2);
%         subplot(2,3,1);plot(log(costs));  
%         subplot(2,3,4);
        display_network(W1');
%         subplot(2,3,5);display_network(W2);

        % 	subplot(2,3,6);imagesc(h);
        
        drawnow
    end
end
