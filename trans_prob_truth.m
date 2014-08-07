trans_prob = zeros(11,11,11,11);
sigma = 1.5;
for cx = 1:11
    for cy = 1:11
        for x = 1:11
            for y = 1:11
                d = sqrt((x-cx)^2+(y-cy)^2);
                trans_prob(cx,cy,x,y) = exp(-d^2/2/sigma^2)/sqrt(2)/sigma;
            end
        end
    end
end
trans_prob = reshape(trans_prob,121,121);
trans_prob = bsxfun(@rdivide, trans_prob, sum(trans_prob(61,:),2));