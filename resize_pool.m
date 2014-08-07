pool_num = size(pool,2);
n = size(pool,1) / 121;
pool1 = reshape(pool,121,n * pool_num);
pool2 = zeros(27*27,n * pool_num);
for i=1:size(pool1,2)
    a = imresize(reshape(pool1(:,i),11,11),[27 27]);
    pool2(:,i)=a(:);
end
pool = reshape(pool2,27*27*n,pool_num);