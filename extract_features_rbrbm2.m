function XC = extract_features_rbrbm2(X, W, bh, rfSize, CIFAR_DIM, M,P)
  whitening = true;
  numCentroids = size(W,1);
  
  % compute features for all training images
  XC = zeros(size(X,1), numCentroids*4);
  for i=1:size(X,1)
    if (mod(i,1000) == 0) fprintf('Extracting features: %d / %d\n', i, size(X,1)); end
    
    % extract overlapping sub-patches into rows of 'patches'
    patches = [ im2col(reshape(X(i,1:1024),CIFAR_DIM(1:2)), [rfSize rfSize]) ;
                im2col(reshape(X(i,1025:2048),CIFAR_DIM(1:2)), [rfSize rfSize]) ;
                im2col(reshape(X(i,2049:end),CIFAR_DIM(1:2)), [rfSize rfSize]) ]';
    patches = 0.2989 * patches(:,1:rfSize^2) + ...
        0.5870 * patches(:,rfSize^2+1:2*rfSize^2) + ...
        0.1140 * patches(:,2*rfSize^2+1:3*rfSize^2);
            
    % do preprocessing for each patch
    
    % normalize for contrast
    patches = bsxfun(@rdivide, bsxfun(@minus, patches, mean(patches,2)), sqrt(var(patches,[],2)+10));
    % whiten
    if (whitening)
      patches = bsxfun(@minus, patches, M) * P;
    end
    
    v = patches';
%     h = W*v;
%     h = bsxfun(@plus, h, bh - sum(W.^2,2)/2);
%     h = bsxfun(@plus, h, -sum(v.^2,1)/2);
%     h = 1 ./ (1 + exp(-h));  % sigmoid
    
    % hard M of K
%     h = sample_MofK(h,sample_1ofK);
    
    % soft sum(h) = 1
%     h = bsxfun(@minus,h,max(h)-1); % normilize
%     h = exp(h);  % sigmoid
%     h = bsxfun(@rdivide, h, sum(h,1));
    
    % triangle
    dist = bsxfun(@plus, bsxfun(@plus, -2*W*v, sum(v.^2,1)), sum(W.^2,2));
    dist = dist.^0.5;
    distm = mean(dist,1);
    distN = bsxfun(@rdivide,dist,distm);
    h = max(0, 1 - 1.3*distN);
      
    % triangle by exp
%     dist = bsxfun(@plus, bsxfun(@plus, -2*W*v, sum(v.^2,1)), sum(W.^2,2));
%     dist = dist.^0.5;
%     distm = mean(dist,1);    
%     distN = bsxfun(@rdivide,dist,distm);
%     h = exp(-4*distN.^2);
%     h = bsxfun(@times, exp(-2*(distN.^2)), distm);
    
    patches = h';    
    % patches is now the data matrix of activations for each patch
    
    % reshape to numCentroids-channel image
    prows = CIFAR_DIM(1)-rfSize+1;
    pcols = CIFAR_DIM(2)-rfSize+1;
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

