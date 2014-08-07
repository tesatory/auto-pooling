function XC = extract_features_ae_nopool_fast(X, W, bh, rfSize, CIFAR_DIM, M,P, stride)
whitening = true;
numCentroids = size(W,1);

prows = length(0:stride:CIFAR_DIM(1)-rfSize);
pcols = length(0:stride:CIFAR_DIM(2)-rfSize);

patches = zeros(rfSize^2*3, size(X,1), prows, pcols);
for offy = 0:stride:CIFAR_DIM(1)-rfSize
    for offx = 0:stride:CIFAR_DIM(2)-rfSize
        a = false(CIFAR_DIM);
        a(offy+(1:rfSize),offx+(1:rfSize),:) = true;
        patches(:,:,offy/stride+1,offx/stride+1) = X(:,a(:))';
    end
end

patches = reshape(patches, rfSize^2*3, size(X,1) * prows * pcols)';
patches = bsxfun(@rdivide, bsxfun(@minus, patches, mean(patches,2)), sqrt(var(patches,[],2)+10));
% whiten
if (whitening)
    patches = bsxfun(@minus, patches, M) * P;
end
patches = 1 ./ (1 + exp(-bsxfun(@plus,  W*patches', bh)));
XC = reshape(patches', size(X,1), numCentroids * prows * pcols);


