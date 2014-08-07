function [patchesA, patchesB] = get_patches_video_file(path, rfSize, scale, data_sz,step)
vr = VideoReader(path);
frame_w = vr.Width;
frame_h = vr.Height;
batchsz = 100; % cant process at once because of low memory

rfSize = rfSize * scale;
patchesA = zeros(rfSize,rfSize,3,data_sz);
patchesB = zeros(rfSize,rfSize,3,data_sz);

% avoid texts in start and end
frame_start = round(vr.NumberOfFrames * 0.2);
frame_end = vr.NumberOfFrames - round(vr.NumberOfFrames * 0.1);

n = 1;
while n <= data_sz    
    fprintf('processing %d\n',n);
    frames = read(vr,[1 batchsz] + randi([frame_start frame_end-batchsz]));
    frames = double(frames) / 256;
    for k = 1:batchsz*10
        % patches at random times
        fn = randi(batchsz - step);
        
        % check continuety
        if mean(mean(mean(abs(frames(:,:,:,fn) ...
                - frames(:,:,:,fn+step))))) > 0.05
            continue
        end
        
        % patches at random positions
        offx = randi([0 frame_w-rfSize]);
        offy = randi([0 frame_h-rfSize]);
        
        A = frames((1:rfSize)+offy,(1:rfSize)+offx,:,fn);
        B = frames((1:rfSize)+offy,(1:rfSize)+offx,:,fn + step);
        
        % check if it is all same color
        if min(std(A(:)), std(B(:))) < 0.04 
            continue
        end
        
        % check if A = B
        if std((A(:)-B(:))) < 0.04
            continue
        end
        
        patchesA(:,:,:,n) = A;
        patchesB(:,:,:,n) = B;
        n = n + 1;
        if n > data_sz
            break
        end
    end
end

rfSize = rfSize / scale;
patchesA2 = zeros(rfSize,rfSize,3,data_sz);
patchesB2 = zeros(rfSize,rfSize,3,data_sz);
for i = 1:data_sz
    patchesA2(:,:,:,i) = imresize(patchesA(:,:,:,i),1/scale);
    patchesB2(:,:,:,i) = imresize(patchesB(:,:,:,i),1/scale);
end
patchesA = reshape(patchesA2, rfSize^2*3, data_sz)';
patchesB = reshape(patchesB2, rfSize^2*3, data_sz)';
