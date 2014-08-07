classdef poolAE
    properties
        lrate = 0.01;
        batchsz = 100;
        bigbatchsz = 10000;
        moment = 0;
        lambda = 1;
        max_moment = 0.9;
        W;
        prec;
    end
    
    methods
        function pool = poolAE(visibleSize, hiddenSize, useSingle)
            if useSingle
                pool.prec = 'single';
            else
                pool.prec = 'double';
            end
            
            pool.W = zeros(visibleSize, hiddenSize, pool.prec);
            r = randperm(visibleSize);
            for h=1:hiddenSize
                pool.W(r(h),h) = 1;
            end
        end
        
        function [cost,grad] = calc_cost(pool, X1, X2)           
            P1 = X1 * pool.W;
            P2 = X2 * pool.W;
            Y1 = P1 * pool.W';
            Y2 = P2 * pool.W';
            
            cost = mean(0.5*sum((X1 - Y1).^2,2) + 0.5*sum((X2 - Y2).^2,2) + ...
                pool.lambda*0.5*sum((P1 - P2).^2,2),1);
            
            grad = (Y1 - X1)' * P1 + X1' * ((Y1 - X1) * pool.W) + ...
                (Y2 - X2)' * P2 + X2' * ((Y2 - X2) * pool.W) + ...
                pool.lambda * (X1 - X2)' * (P1 - P2);
            
            grad = grad / size(X1,1);
        end
        
        function pool = train(pool, X1, X2, max_epoch, useGPU)
            data_sz = size(X1,1);
            pool.moment = 0;            
            speed = zeros(size(pool.W), pool.prec);
            
            if isa(pool.W, 'single') && isa(X1, 'double')
                X1 = single(X1);
                X2 = single(X2);
            end
            
            % shuffle train data
            r = randperm(data_sz);
            X1 = X1(r,:);
            X2 = X2(r,:);
            
            if useGPU
                pool.W = gpuArray(pool.W);
                speed = gpuArray(speed);
                %pool.W = gsingle(pool.W);
                %speed = gsingle(speed);
            else
                pool.bigbatchsz = data_sz;
            end
            
            tic;
            for epoch = 1:max_epoch
                if epoch == 5
                    pool.moment = pool.max_moment;
                end
                cost = 0;
                
                % when using GPU, only subset of train data can fit in
                % devide memory
                for nn = (1:data_sz / pool.bigbatchsz)
                    if useGPU
                        clear X1gpu X2gpu
                        X1gpu = gpuArray(X1(1+pool.bigbatchsz*(nn-1):pool.bigbatchsz*nn,:));
                        X2gpu = gpuArray(X2(1+pool.bigbatchsz*(nn-1):pool.bigbatchsz*nn,:));
                        %X1gpu = gsingle(X1(1+pool.bigbatchsz*(nn-1):pool.bigbatchsz*nn,:));
                        %X2gpu = gsingle(X2(1+pool.bigbatchsz*(nn-1):pool.bigbatchsz*nn,:));
                    end
                    for n = (1:pool.bigbatchsz / pool.batchsz)
                        if useGPU
                            X1_sub = X1gpu(1+pool.batchsz*(n-1):pool.batchsz*n,:);
                            X2_sub = X2gpu(1+pool.batchsz*(n-1):pool.batchsz*n,:);
                        else
                            X1_sub = X1(1+pool.batchsz*(n-1):pool.batchsz*n,:);
                            X2_sub = X2(1+pool.batchsz*(n-1):pool.batchsz*n,:);
                        end
                        
                        [c, grad] = pool.calc_cost(X1_sub, X2_sub);
                        cost = cost + c;
                        
                        speed = speed * pool.moment - grad;
                        pool.W = pool.W + pool.lrate * speed;
                        pool.W = pool.W .* (pool.W > 0); % keep positive
                    end
                end
                cost = cost * pool.batchsz / data_sz;
                fprintf(1,'epoch %d \t cost %f\n',epoch, cost);
            end
            toc;
            
            if useGPU
                clear X1gpu X2gpu;
                pool.W = gather(pool.W);
            end
        end
    end
end