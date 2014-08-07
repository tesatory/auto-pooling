function [WA,WB] = divide_region_by_line(Worig, sz)
WorigN = reshape(Worig/sum(Worig(:)),sz,sz);
midX = sum(sum(WorigN .* repmat((1:sz),sz,1)));
midY = sum(sum(WorigN .* repmat((1:sz)',1,sz)));
WA = zeros(100,sz^2);
WB = zeros(100,sz^2);
for i = 1:100
    a1 = zeros(sz,sz);
    a2 = zeros(sz,sz);
    r = pi*i/100;

    for y = 1:sz
        for x = 1:sz
            if (x-midX) * sin(r) < (y-midY) * cos(r)
                a1(y,x) = 1;
            end
            if (x-midX) * sin(r) <= (y-midY) * cos(r)
                a2(y,x) = 1;
            end
        end
    end
    if abs(sum((2*a1(:)'-1) .* Worig)) < abs(sum((2*a2(:)'-1) .* Worig))
        WA(i,:) = a1(:)' .* Worig;
        WB(i,:) = (1 - a1(:)') .* Worig;
    else
        WA(i,:) = a2(:)' .* Worig;
        WB(i,:) = (1 - a2(:)') .* Worig;
    end
end
WA = bsxfun(@rdivide,WA,sqrt(sum(WA.^2,2))); % normalize to length 1
WB = bsxfun(@rdivide,WB,sqrt(sum(WB.^2,2))); % normalize to length 1