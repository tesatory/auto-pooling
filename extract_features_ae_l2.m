function [XC,XCP] = extract_features_ae_l2(X,W1,b1)
% X : 8100,N

M = size(W1,1);
N = size(X,2);
X = reshape(X, 9, 9, 100, N);
XC = zeros(N,M,7,7);
for y = 1:7
    for x = 1:7
        XP = X((1:3)+(y-1),(1:3)+(x-1),:,:);
        XP = reshape(XP,900,N);
        XP = bsxfun(@rdivide,bsxfun(@minus,XP,mean(XP,1)),sqrt(var(XP,[],1)+0.1));
        XP = bsxfun(@plus, W1 * XP, b1);
        XP = 1 ./ (1 + exp(-XP));
        XC(:,:,y,x) = XP';
    end
end

XCP = zeros(N,M,4);
XCP(:,:,1) = sum(sum(XC(:,:,1:4,1:4),4),3);
XCP(:,:,2) = sum(sum(XC(:,:,1:4,5:7),4),3);
XCP(:,:,3) = sum(sum(XC(:,:,5:7,1:4),4),3);
XCP(:,:,4) = sum(sum(XC(:,:,5:7,5:7),4),3);
XCP = reshape(XCP,N,M*4)';
XC = reshape(XC,N,M*49)';

end