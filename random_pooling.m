function pool = random_pooling(N, N2, centroids, sumdata, rmin, rmax, corr_min)
sz = 27;
M = size(centroids,1);
pool = zeros(sz,sz,M,N);
fcorr = corr(sumdata);
gsz = zeros(N,1);
for n = 1:N
    cx = randi([1 sz]);
    cy = randi([1 sz]);
    r = randi([rmin rmax]);
    
    f = randi(M);
    v = fcorr(f,:);
    
    for m = 1:M
        if v(m) >= corr_min
            gsz(n) = gsz(n) + 1;
            for y = 1:sz
                for x = 1:sz
                    if (x-cx)^2 + (y-cy)^2 < r^2
                        pool(y,x,m,n) = 1;
                    end
                end
            end
        end
    end
end
pool = reshape(pool, sz^2 * M, N);

a = pool' * pool;
b = bsxfun(@plus, sum(pool)', sum(pool));
b = b - a;
a = a ./ b;
a = a - diag(diag(a));
for n = 1:N - N2
    [x,i] = max(a);
    [x,j] = max(x);
    pool(:,j) = 0;
    a(:,j) = 0;
    a(j,:) = 0;
end
[xx,y] = sort(sum(pool),'descend');
pool = pool(:,y(1:N2));

fprintf('max overlap %f\n',x);
fprintf('average group size %f\n',mean(gsz));
