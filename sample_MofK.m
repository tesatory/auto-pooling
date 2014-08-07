function hval = sample_MofK(h,M)
hval = false(size(h));
for m = 1:M
    hval = hval | sample_1ofK(h);
    for i = 1:size(h,2)
        h(hval(:,i),i) = - inf; % prevent selection of the same node
    end
end