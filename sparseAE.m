classdef sparseAE
    properties
        lrate = 0.1;
        batchsz = 1000;
        moment = 0;
        sparsityParam = 0.035; % desired average activation of the hidden units.
        lambda = 3e-3;         % weight decay parameter
        beta = 5;              % weight of sparsity penalty term
        max_moment = 0.9;
        W1; W2; b1; b2;
        prec;
    end
    
    methods
        function ae = sparseAE(visibleSize, hiddenSize, useSingle)
            if useSingle
                ae.prec = 'single';
            else
                ae.prec = 'double';
            end
            
            r  = sqrt(6) / sqrt(hiddenSize+visibleSize+1);   % we'll choose weights uniformly from the interval [-r, r]
            ae.W1 = rand(hiddenSize, visibleSize, ae.prec) * 2 * r - r;
            ae.W2 = rand(visibleSize, hiddenSize, ae.prec) * 2 * r - r;
            ae.b1 = zeros(hiddenSize, 1, ae.prec);
            ae.b2 = zeros(visibleSize, 1, ae.prec);
        end
        
        function [cost,W1grad,W2grad,b1grad,b2grad] = calc_cost(ae, data)           
            a1 = bsxfun(@plus, ae.W1 * data, ae.b1);
            h1 = 1./(1 + exp(-a1));
            a2 = bsxfun(@plus, ae.W2 * h1, ae.b2);
            h2 = a2;
            
            cost = mean(sum(0.5*(h2 - data).^2,1),2);
            cost_sparse = ae.sparsityParam * log(ae.sparsityParam ./ mean(h1,2)) + ...
                (1 - ae.sparsityParam) * log((1-ae.sparsityParam) ./ (1-mean(h1,2)));
            cost = cost + ae.beta * sum(cost_sparse);
            
            datasz = size(data,2);
            t2 = (h2 - data)/ datasz;
            b2grad = sum(t2,2);
            W2grad = t2 * h1';
            
            t1 = ae.W2' * t2;
            t3 = ae.beta * (- ae.sparsityParam ./ mean(h1,2) + ...
                (1 - ae.sparsityParam) ./ (1 - mean(h1,2))) / datasz;
            t1 = bsxfun(@plus, t1, t3);
            t1 = t1 .* h1 .* (1 - h1);
            b1grad = sum(t1,2);
            W1grad = t1 * data';
            
            % add weight regularization cost
            cost = cost + ae.lambda*0.5*(sum(sum(ae.W1.^2)) + sum(sum(ae.W2.^2)));
            W1grad = W1grad + ae.lambda * ae.W1;
            W2grad = W2grad + ae.lambda * ae.W2;
        end
        
        function ae = train(ae, data, max_epoch, useGPU)
            data_sz = size(data,2);
            ae.moment = 0;
            W1speed = zeros(size(ae.W1), ae.prec);
            W2speed = zeros(size(ae.W2), ae.prec);
            b1speed = zeros(size(ae.b1), ae.prec);
            b2speed = zeros(size(ae.b2), ae.prec);
            
            if isa(ae.W1, 'single') && isa(data, 'double')
                data = single(data);
            end
            
            % shuffle train data
            r = randperm(data_sz);
            data = data(:,r);

            if useGPU
                data = gpuArray(data);
                ae.W1 = gpuArray(ae.W1);
                ae.W2 = gpuArray(ae.W2);
                ae.b1 = gpuArray(ae.b1);
                ae.b2 = gpuArray(ae.b2);
                W1speed = gpuArray(W1speed);
                W2speed = gpuArray(W2speed);
                b1speed = gpuArray(b1speed);
                b2speed = gpuArray(b2speed);
            end
            
            tic;
            for epoch = 1:max_epoch
                if epoch == 5
                    ae.moment = ae.max_moment;
                end
                cost = 0;

                for n = (1:data_sz / ae.batchsz)
                    batch = data(:,1+ae.batchsz*(n-1):ae.batchsz*n);
                        
                    [c, W1grad, W2grad, b1grad, b2grad] = ae.calc_cost(batch);
                    cost = cost + c;
                        
                    W1speed = W1speed * ae.moment - W1grad;
                    W2speed = W2speed * ae.moment - W2grad;
                    b1speed = b1speed * ae.moment - b1grad;
                    b2speed = b2speed * ae.moment - b2grad;
                    ae.W1 = ae.W1 + ae.lrate * W1speed;
                    ae.W2 = ae.W2 + ae.lrate * W2speed;
                    ae.b1 = ae.b1 + ae.lrate * b1speed;
                    ae.b2 = ae.b2 + ae.lrate * b2speed;
                end
                cost = cost * ae.batchsz / data_sz;
                fprintf(1,'epoch %d \t cost %f\n',epoch, cost);
            end
            toc;
            
            if useGPU
                ae.W1 = gather(ae.W1);
                ae.W2 = gather(ae.W2);
                ae.b1 = gather(ae.b1);
                ae.b2 = gather(ae.b2);
            end
        end
    end
end