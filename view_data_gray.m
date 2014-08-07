function img = view_data_gray(data,w,h,xrep,yrep)
N = size(data,1);
img = zeros(h * yrep, w * xrep);
c = 1;
for y = 1:yrep
    for x = 1:xrep
        if c <= N
            img((1:h)+h*(y-1),(1:w)+w*(x-1)) = reshape(data(c,:),h,w);
        end
        c = c+1;
    end
end
% img = (img - min(img(:))) / (max(img(:)) - min(img(:)));
imagesc(img);
colormap(gray);