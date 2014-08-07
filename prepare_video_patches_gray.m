function [patchesA, patchesB, M, P] = prepare_video_patches_gray(rfSize, scale, data_sz,step)
vr = VideoReader('video/bbc_nature.mp4');
frame_w = vr.Width;
frame_h = vr.Height;
batchsz = 100; % cant process at once because of low memory

% scale = 4;
rfSize = rfSize * scale;
patchesA = zeros(rfSize,rfSize,data_sz);
patchesB = zeros(rfSize,rfSize,data_sz);

n = 1;
while n <= data_sz    
    fprintf('processing %d\n',n);
    frames = read(vr,[1 batchsz] + randi([0 vr.NumberOfFrames-batchsz]));
    frames_gray = zeros(frame_h, frame_w, batchsz);
    for i = 1:batchsz
        frames_gray(:,:,i) = rgb2gray(frames(:,:,:,i));
    end
    frames_gray = frames_gray / 256;
    for k = 1:batchsz
        % patches at random times
        fn = randi(batchsz - step);
        
        % check continuety
        if mean(mean(abs(frames_gray(:,:,fn) ...
                - frames_gray(:,:,fn+step)))) < 0.05
            % patches at random positions
            offx = randi([0 frame_w-rfSize]);
            offy = randi([0 frame_h-rfSize]);
            
            A = frames_gray((1:rfSize)+offy,(1:rfSize)+offx,fn);
            B = frames_gray((1:rfSize)+offy,(1:rfSize)+offx,fn + step);
            
            % check if it is all same color
            if min(std(A(:)), std(B(:))) > 0.03 
                patchesA(:,:,n) = A;
                patchesB(:,:,n) = B;
                n = n + 1;
                if n > data_sz
                    break
                end
            end
        end
    end

end
rfSize = rfSize / scale;
patchesA2 = zeros(rfSize,rfSize,data_sz);
patchesB2 = zeros(rfSize,rfSize,data_sz);
for i = 1:data_sz
    patchesA2(:,:,i) = imresize(patchesA(:,:,i),1/scale);
    patchesB2(:,:,i) = imresize(patchesB(:,:,i),1/scale);
end
patchesA = reshape(patchesA2, rfSize^2, data_sz)';
patchesB = reshape(patchesB2, rfSize^2, data_sz)';

%whiten 
C = cov(patchesA);
M = mean(patchesA);
[V,D] = eig(C);
P = V * diag(sqrt(1./(diag(D) + 0.001))) * V';
% patchesA = bsxfun(@minus,patchesA,M) * P;
% patchesB = bsxfun(@minus,patchesB,M) * P;

end