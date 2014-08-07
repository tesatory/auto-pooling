prepare_cifar10

addpath minFunc;
rfSize = 6;
whitening=true;
numPatches = 400000;
CIFAR_DIM=[32 32 3];

% extract random patches
patches = zeros(numPatches, rfSize*rfSize*3);
for i=1:numPatches
  if (mod(i,10000) == 0) fprintf('Extracting patch: %d / %d\n', i, numPatches); end
  
  r = random('unid', CIFAR_DIM(1) - rfSize + 1);
  c = random('unid', CIFAR_DIM(2) - rfSize + 1);
  patch = reshape(trainX(mod(i-1,size(trainX,1))+1, :), CIFAR_DIM);
  patch = patch(r:r+rfSize-1,c:c+rfSize-1,:);
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

for numCentroids = [100 200 400 800 1200 1600] 
    % run K-means
    centroids = run_kmeans(patches, numCentroids, 50);
    save(strcat('centroids_',int2str(numCentroids)), 'centroids', 'CIFAR_DIM', 'rfSize', 'M', 'P');
end