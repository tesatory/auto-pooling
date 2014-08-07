function pool = resize_pool_11to14(pool)
pool_num = size(pool,2);
n = size(pool,1) / 121;
pool = reshape(pool,121,n * pool_num);
W = zeros(14,14,11,11);
for y = 0:13
    for x = 0:13
        xx = 10*x/13;
        yy = 10*y/13;
        x1 = floor(xx);
        y1 = floor(yy);
        x2 = ceil(xx);
        y2 = ceil(yy);
        xd = xx - x1;
        yd = yy - y1;
        W(y+1,x+1,y1+1,x1+1) = W(y+1,x+1,y1+1,x1+1) + (1-xd) * (1- yd);
        W(y+1,x+1,y2+1,x1+1) = W(y+1,x+1,y2+1,x1+1) + (1-xd) * (yd);
        W(y+1,x+1,y1+1,x2+1) = W(y+1,x+1,y1+1,x2+1) + (xd) * (1- yd);
        W(y+1,x+1,y2+1,x2+1) = W(y+1,x+1,y2+1,x2+1) + (xd) * (yd);
    end
end
W = reshape(W, 14*14,11*11);

pool = W * pool;
pool = reshape(pool, 14*14*n, pool_num);