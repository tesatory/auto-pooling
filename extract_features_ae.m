function XC = extract_features_ae(X, W, bh, rfSize, CIFAR_DIM, M,P,stride)
  whitening = stride;
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
  XC = zeros(size(X,1), numCentroids*4);
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
    
    % reshape to numCentroids-channel image
    patches = reshape(patches, prows, pcols, numCentroids);
    
    % pool over quadrants
    halfr = round(prows/2);
    halfc = round(pcols/2);
    q1 = sum(sum(patches(1:halfr, 1:halfc, :), 1),2);
    q2 = sum(sum(patches(halfr+1:end, 1:halfc, :), 1),2);
    q3 = sum(sum(patches(1:halfr, halfc+1:end, :), 1),2);
    q4 = sum(sum(patches(halfr+1:end, halfc+1:end, :), 1),2);
    
    % concatenate into feature vector
    XC(i,:) = [q1(:);q2(:);q3(:);q4(:)]';
  end

