function XC = extract_features_ae_pooled(X, W, bh, pool, rfSize, CIFAR_DIM, M,P,stride)
  whitening = true;
  numCentroids = size(W,1);

  prows = CIFAR_DIM(1)-rfSize+1;
  pcols = CIFAR_DIM(2)-rfSize+1;
  if stride > 1
      a = false(prows, pcols);
      l = 1:stride:prows;
      a(l,l) = true;
      prows = length(l);
      pcols = length(l);
  end
  dim = CIFAR_DIM(1) * CIFAR_DIM(2);

  % compute features for all training images
  XC = zeros(size(X,1), size(pool,2));
  for i=1:size(X,1)
    if (mod(i,1000) == 0) fprintf('Extracting features: %d / %d\n', i, size(X,1)); end
    
    % extract overlapping sub-patches into rows of 'patches'
    if CIFAR_DIM(3) == 3
        patches = [ im2col(reshape(X(i,1:dim),CIFAR_DIM(1:2)), [rfSize rfSize]) ;
            im2col(reshape(X(i,dim+1:2*dim),CIFAR_DIM(1:2)), [rfSize rfSize]) ;
            im2col(reshape(X(i,2*dim+1:end),CIFAR_DIM(1:2)), [rfSize rfSize]) ]';
    else
        patches = im2col(reshape(X(i,1:dim),CIFAR_DIM(1:2)), [rfSize rfSize])';
    end
    
    % stride
    if stride > 1
        patches = patches(a(:),:);
    end
    
    % do preprocessing for each patch
    
    % normalize for contrast
    patches = bsxfun(@rdivide, bsxfun(@minus, patches, mean(patches,2)), sqrt(var(patches,[],2)+10));
    % whiten
    if (whitening)
      patches = bsxfun(@minus, patches, M) * P;
    end
    
    v = patches';
    h = bsxfun(@plus,  W*v, bh);
    h = 1 ./ (1 + exp(-h));  % sigmoid
    patches = h';    
    % patches is now the data matrix of activations for each patch
    
    % concatenate into feature vector
    XC(i,:) = patches(:)' * pool;
  end

