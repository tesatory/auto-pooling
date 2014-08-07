function W = toy_pooling1(sz)
W = zeros(100,sz^2);
for i = 1:100
    a = zeros(sz,sz);
    for y = 1:sz
        for x = 1:sz
            b = 2*pi*i/100;
            if (x-sz/2) * sin(b) < (y-sz/2) * cos(b)
                a(y,x) = 1;
            end
        end
    end
    W(i,:) = a(:)';
end
