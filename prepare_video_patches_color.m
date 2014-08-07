function [patchesA, patchesB] = prepare_video_patches_color(rfSize, scale, data_sz,step)
vr = VideoReader('video\bbc_nature.mp4');
frame_w = vr.Width;
frame_h = vr.Height;
batchsz = 100; % cant process at once because of low memory

% scale = 4;
rfSize = rfSize * scale;
patchesA = zeros(rfSize,rfSize,3,data_sz);
patchesB = zeros(rfSize,rfSize,3,data_sz);

n = 1;
while n <= data_sz    
    fprintf('processing %d\n',n);
    frames = read(vr,[1 batchsz] + randi([0 vr.NumberOfFrames-batchsz]));
    frames = double(frames);
    for k = 1:batchsz
        % patches at random times
        fn = randi(batchsz - step);
        
        % check continuety
        if mean(mean(mean(abs(frames(:,:,:,fn) ...
                - frames(:,:,:,fn+step))))) < 0.05 * 256
            % patches at random positions
            offx = randi([0 frame_w-rfSize]);
            offy = randi([0 frame_h-rfSize]);
            
            A = frames((1:rfSize)+offy,(1:rfSize)+offx,:,fn);
            B = frames((1:rfSize)+offy,(1:rfSize)+offx,:,fn + step);
            
            % check if it is all same color
            if min(std(A(:)), std(B(:))) > 0.06 * 256 && ...
                    mean((A(:)-B(:)).^2) > 50
                patchesA(:,:,:,n) = A;
                patchesB(:,:,:,n) = B;
                n = n + 1;
                if n > data_sz
                    break
                end
            end
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

% patchesA = bsxfun(@rdivide, bsxfun(@minus, patchesA, mean(patchesA,2)), sqrt(var(patchesA,[],2)+10));
% patchesB = bsxfun(@rdivide, bsxfun(@minus, patchesB, mean(patchesB,2)), sqrt(var(patchesB,[],2)+10));
end