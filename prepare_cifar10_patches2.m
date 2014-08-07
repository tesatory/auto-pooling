prepare_cifar10

addpath minFunc;
rfSize = 10;
whitening=true;
numPatches = 100000;
CIFAR_DIM=[32 32 3];

% extract random patches
patches = zeros(numPatches, rfSize*rfSize);
for i=1:numPatches
  if (mod(i,10000) == 0) fprintf('Extracting patch: %d / %d\n', i, numPatches); end
  
  r = random('unid', CIFAR_DIM(1) - rfSize + 1);
  c = random('unid', CIFAR_DIM(2) - rfSize + 1);
  patch = reshape(trainX(mod(i-1,size(trainX,1))+1, :), CIFAR_DIM);
  patch = 0.2989 * patch(r:r+rfSize-1,c:c+rfSize-1,1) + ...
      0.5870 * patch(r:r+rfSize-1,c:c+rfSize-1,2) + ...
      0.1140 * patch(r:r+rfSize-1,c:c+rfSize-1,3);
  patches(i,:) = patch(:)';
end

% normalize for contrast
patches = bsxfun(@rdivide, bsxfun(@minus, patches, mean(patches,2)), sqrt(var(patches,[],2)+10));

% whiten
if (whitening)
  C = cov(patches);
  M = mean(patches);
  [V,D] = eig(C);
  P = V * diag(sqrt(1./(diag(D) + 0.1))) * V';
  patches = bsxfun(@minus, patches, M) * P;
end
