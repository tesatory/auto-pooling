function W = auto_pooling(data1, data2, out_dim, epoch, useGPU, useSingle, W)
assert(nargin == 6 || nargin == 7);
in_dim = size(data1,2);

if nargin == 6
    W = 1 + 0.2*randn(out_dim, in_dim);
    W = bsxfun(@rdivide, W, sum(W,1)); % normilize to cover all area
    W = bsxfun(@rdivide, W, sqrt(sum(W.^2,2))); % normilize by length
%     W = zeros(out_dim, in_dim);
%     for i=1:out_dim
%         W(i,randi(in_dim)) = 1;
%     end
end

batch_sz = 1000;
lrate = 0.01;
dataD = data1 - data2;
dataD = bsxfun(@rdivide,dataD,std(dataD')' + 0.00001); % normalize

if useSingle
    data1 = single(data1);
    data2 = single(data2);
    dataD = single(dataD);
    W = single(W);
    lrate = single(lrate);
    batch_sz = single(batch_sz);
end

if useGPU
    data1 = gpuArray(data1);
    data2 = gpuArray(data2);
    dataD = gpuArray(dataD);
    W = gpuArray(W);
end

for t = 1:epoch
    if mod(t,10) == 1
        for i=1:size(W,1);
            view_data_gray(W,sqrt(size(W,2)),sqrt(size(W,2)),size(W,1),1);
        end
        pause(0.1);
    end
    
    if mod(t,10) == 1
        s = W * (data1 - data2)';
        s = sum(abs(s),1);
%         s = sum(1./(1+exp(-abs(s))),1);
        s = mean(s);
        
        p = W * (data1(randperm(size(data1,1)),:) - data2)';
        p = sum(abs(p),1);
%         p = sum(1./(1+exp(-abs(p))),1);
        p = mean(p);
        fprintf('epoch: %d dis:%f / %f = %f\n', t, s, p, s/p);
    end
    
    for k = 1:size(data1,1)/batch_sz
        c = randi(size(data1,1),batch_sz,1);        
        d = randi(size(data1,1),batch_sz,1);
        data = dataD(c,:);
        W = W - (lrate/batch_sz) * sign(W * data') * data;  % grad desc by sum of dist in 1 dim
%         X = mean(sqrt(sum((W *data').^2,1)));
%         Xder = bsxfun(@rdivide, (W *data'), sqrt(sum((W *data').^2,1)) + 0.00000001) * data / batch_sz;
        data = data1(c,:)-data2(d,:);
        W = W + (lrate/batch_sz) * sign(W * data') * data;  % grad desc by sum of dist in 1 dim
%         Y = mean(sqrt(sum((W *data').^2,1)));
%         Yder = bsxfun(@rdivide, (W *data'), sqrt(sum((W *data').^2,1)) + 0.00000001) * data / batch_sz;
%         W = W - (lrate/batch_sz) * Xder;
%         W = W - (lrate/batch_sz) * (Xder * Y - X * Yder) / Y^2;

%         W = W - (lrate/batch_sz) * (W * data') * data; % grad desc by sum of sq dist

        %         W = W - (lrate/batch_sz) * (sign(W * data') ./ (abs(W * data')).^0.5) * data;  % grad desc by sum of dist in 1 dim
%         W = W - (lrate/batch_sz) * bsxfun(@rdivide, W * data', sum((W * data').^2,1).^0.5 + 0.001) * data;  % grad desc by sum of dist 
%         W = W - (lrate/batch_sz) * bsxfun(@times, sign(W * data') ./ sqrt(abs(W * data') + 0.00001), sum(abs(W * data').^0.5,1)) * data;  % grad desc by sum of dist 
%         W = W - (lrate/batch_sz) * (sign(W * data') ./ sqrt(abs(W * data') + 0.00001)) * data;  % grad desc by sum of dist 

%         temp = 0.1;
%         y = 1 ./ (1 + exp(-abs(W * data' / temp)));
%         W = W - (lrate/batch_sz) * (y .* (1-y) .* sign(W * data')) * data / temp;
        
        W = W .* (W > 0); % keep positive
%         W = bsxfun(@rdivide, W, sum(W,2)); % normilize
%         W = bsxfun(@rdivide, W, sum(W,1)); % normilize to cover all area
%         W = bsxfun(@rdivide, W, sqrt(sum(W.^2,2))); % normilize by length
%         W = bsxfun(@rdivide, W, sqrt(sum(W.^2,1))); % normilize 
    end    
end