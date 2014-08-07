load work\121211_vimio_full_100k_single.mat
dataAs = zeros(100000,256);
dataBs = zeros(100000,256);
for k = 1:100000
    a = reshape(dataA(r(k),:),32,32,3);
    as = rgb2gray(a);
    as = as(1:16,1:16);
    dataAs(k,:) = as(:)';
    b = reshape(dataB(r(k),:),32,32,3);
    bs = rgb2gray(b);
    bs = bs(1:16,1:16);
    dataBs(k,:) = bs(:)';
end
dataAs = dataAs*256;
dataBs = dataBs*256;
dataAm = bsxfun(@rdivide, bsxfun(@minus, dataAs, mean(dataAs,2)), sqrt(var(dataAs,[],2)+10));
dataBm = bsxfun(@rdivide, bsxfun(@minus, dataBs, mean(dataBs,2)), sqrt(var(dataBs,[],2)+10));
dataAw = bsxfun(@minus, dataAm, M) * P;
dataBw = bsxfun(@minus, dataBm, M) * P;