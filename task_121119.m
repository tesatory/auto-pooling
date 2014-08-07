%pool = zeros(121,4,100);
%hdataA = extract_features_ae_nopool(dataA,W1,b1,rfSize,[16 16 3],M,P,1);
%hdataB = extract_features_ae_nopool(dataB,W1,b1,rfSize,[16 16 3],M,P,1);
for k = 1:100
    fprintf('feature %d\n', k);
    hdataA2 = hdataA(:,121*k-120:121*k);
    hdataB2 = hdataB(:,121*k-120:121*k);
    pool(:,:,k) = train_pool_ae(hdataA2,hdataB2,4,300);
end
