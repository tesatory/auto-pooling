function inhib_cost = get_inhib_cost(w,h,s)
inhib_cost = zeros(w*h);
c = 1;
for y = 1:h
    for x = 1:w
        cc = 1;
        for yy = 1:h
            for xx = 1:w
                d = sqrt((x - xx)^2 + (y -yy)^2);
                inhib_cost(cc,c) = exp(-d^2/s);
                cc = cc + 1;
            end
        end
        c = c + 1;
    end
end

display_network(inhib_cost);
