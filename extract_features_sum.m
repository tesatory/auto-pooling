function hdata = extract_features_sum(patches, centroids, rfSize, CIFAR_DIM)
prows = CIFAR_DIM(1)-rfSize+1;
pcols = CIFAR_DIM(2)-rfSize+1;
  
% compute 'triangle' activation function
xx = sum(patches.^2, 2);
cc = sum(centroids.^2, 2)';
xc = patches * centroids';

z = sqrt( bsxfun(@plus, cc, bsxfun(@minus, xx, 2*xc)) ); % distances
mu = mean(z, 2); % average distance to centroids for each patch
patches = max(bsxfun(@minus, mu, z), 0);
% patches is now the data matrix of activations for each patch
patches = reshape(patches, size(patches,1)/prows/pcols, prows*pcols, size(centroids,1));
hdata = reshape(sum(patches,2), size(patches,1), size(centroids,1));