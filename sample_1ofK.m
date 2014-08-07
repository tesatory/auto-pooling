function hval = sample_1ofK(h)
h = bsxfun(@minus,h,max(h)-1); % normilize
h = exp(h);  % sigmoid
a = 0;
while true
    r = rand(1,size(h,2)) .* sum(h,1);
    hval = false(size(h));
    for i = 1:size(h,1)
        x = (r > 0);
        r = r - h(i,:);
        x = x & (r <= 0);
        hval(i,x) = true;
    end
    if sum(abs(sum(hval,1)-1)) == 0
        break
    end
    a = a + 1;
    if a > 100
        keyboard
    end
end