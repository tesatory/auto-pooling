function pool = resize_pool_fast(pool)
pool_num = size(pool,2);
n = size(pool,1) / 121;
pool = reshape(pool,121,n * pool_num);
W = zeros(27,27,11,11);
for y = 0:26
    for x = 0:26
        xx = 10*x/26;
        yy = 10*y/26;
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
W = reshape(W, 27*27,11*11);

pool = W * pool;
pool = reshape(pool, 27*27*n, pool_num);