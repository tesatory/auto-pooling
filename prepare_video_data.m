[data1,diff1]=get_small_img_from_video('video/tokyo.mp4',32,1,0);
[data2,diff2]=get_small_img_from_video('video/bbc_nature.mp4',32,1,0);
[data3,diff3]=get_small_img_from_video('video/bbc_horizon.mp4',32,1,0);
[data4,diff4]=get_small_img_from_video('video/hot_planet.mp4',32,1,0);
% [data5,diff5]=get_small_img_from_video('video/human_planet.mp4',32,1,0);
% [data6,diff6]=get_small_img_from_video('video/reindeer.mp4',32,1,0);

data1 = (data1 - min(data1(:))) / (max(data1(:)) - min(data1(:)));
data2 = (data2 - min(data2(:))) / (max(data2(:)) - min(data2(:)));
data3 = (data3 - min(data3(:))) / (max(data3(:)) - min(data3(:)));
data4 = (data4 - min(data4(:))) / (max(data4(:)) - min(data4(:)));

data = [data1; data2; data3; data4];
diff = [diff1; diff2; diff3; diff4];

% reduce noise
dataM = bsxfun(@minus, data, mean(data,1));
[W,E] = eig(cov(dataM));
W = W(:,257:end);
data = data * (W * W');

maxdiff = 100;
mindiff = 40;
c = 0;
dataA = zeros(size(data));
dataB = zeros(size(data));
for i = 2:size(data,1)
    if diff(i,1) > mindiff && diff(i,1) < maxdiff
        c = c + 1;
        dataA(c,:) = data(i-1,:);
        dataB(c,:) = data(i,:);
    end
end
dataA = dataA(1:c,:);
dataB = dataB(1:c,:);
