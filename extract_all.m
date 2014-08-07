function XC = extract_all(X, centroids, rfSize, CIFAR_DIM, M,P)
  assert(nargin == 4 || nargin == 6);
  whitening = (nargin == 6);
  numCentroids = size(centroids,1);
  
  dim = CIFAR_DIM(1) * CIFAR_DIM(2);
  % compute features for all training images
  prows = CIFAR_DIM(1)-rfSize+1;
  pcols = CIFAR_DIM(2)-rfSize+1;

  XC = zeros(size(X,1), numCentroids*prows*pcols);
  for i=1:size(X,1)
    if (mod(i,1000) == 0) fprintf('Extracting features: %d / %d\n', i, size(X,1)); end
    
    % extract overlapping sub-patches into rows of 'patches'
    patches = [ im2col(reshape(X(i,1:dim),CIFAR_DIM(1:2)), [rfSize rfSize]) ;
                im2col(reshape(X(i,dim+1:2*dim),CIFAR_DIM(1:2)), [rfSize rfSize]) ;
                im2col(reshape(X(i,2*dim+1:end),CIFAR_DIM(1:2)), [rfSize rfSize]) ]';

    % do preprocessing for each patch
    
    % normalize for contrast
    patches = bsxfun(@rdivide, bsxfun(@minus, patches, mean(patches,2)), sqrt(var(patches,[],2)+10));
    % whiten
    if (whitening)
      patches = bsxfun(@minus, patches, M) * P;
    end
    
    % compute 'triangle' activation function
    xx = sum(patches.^2, 2);
    cc = sum(centroids.^2, 2)';
    xc = patches * centroids';
    
    z = sqrt( bsxfun(@plus, cc, bsxfun(@minus, xx, 2*xc)) ); % distances
    [v,inds] = min(z,[],2);
    mu = mean(z, 2); % average distance to centroids for each patch
    patches = max(bsxfun(@minus, mu, z), 0);
    % patches is now the data matrix of activations for each patch
    
    % reshape to numCentroids-channel image
    patches = reshape(patches, prows, pcols, numCentroids);
    
    % concatenate into feature vector
    XC(i,:) = patches(:)';
  end
