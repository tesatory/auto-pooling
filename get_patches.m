function patches = get_patches(X, rfSize, CIFAR_DIM, M, P)
% extract small patches from images
prows = CIFAR_DIM(1)-rfSize+1;
pcols = CIFAR_DIM(2)-rfSize+1;

patch_indx = zeros(prows * pcols,size(X,2));
i = 1;
for r = 1:prows
    for c = 1:pcols
        a = zeros(CIFAR_DIM);
        a((1:rfSize)+c-1,(1:rfSize)+r-1,:) = 1;
        patch_indx(i,:) = a(:)';
        i = i + 1;
    end
end

patches = zeros(size(X,1), size(patch_indx,1), rfSize * rfSize * 3);
for j = 1:size(patch_indx,1)
    patches(:,j,:) = X(:,patch_indx(j,:)==1);
end

patches = reshape(patches, size(patch_indx,1) * size(X,1), rfSize * rfSize * 3);

% normalize for contrast
patches = bsxfun(@minus, patches, mean(patches,2));
patches = bsxfun(@rdivide, patches, sqrt(var(patches,[],2)+10));
% whiten
patches = bsxfun(@minus, patches, M) * P;
