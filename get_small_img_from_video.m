function [data,diff] = get_16x16_from_video_color(path, sz, step, offset)
vr = VideoReader(path);
fnum = floor(vr.NumberOfFrames/step);
fnum = min(fnum, 20000);
frame_w = vr.Width;
frame_h = vr.Height;
% offx = floor((frame_w - 320)/2);
% offy = floor((frame_h - 320)/2);
offx = floor((frame_w - frame_h)/2);
offy = 1;
batchsz = 100; % cant process at once because of low memory
frames_small = zeros(sz,sz,3,fnum);
diff = zeros(fnum, 1);
diff(1,1) = 256;
for fsta = 1:batchsz:fnum
    frames = uint8(zeros(frame_h, frame_w, 3, min(batchsz,fnum-fsta+1)));
    for i = 1:size(frames,4)
        frames(:,:,:,i) = read(vr,(i+fsta-2)*step+1+offset);
        
        if fsta+i-1 > 1
            if i == 1
                a = last_frame - frames(:,:,:,1);
            else
                a = frames(:,:,:,i-1) - frames(:,:,:,i);
            end
            diff(fsta+i-1,1) = mean(a(:).^2);
        end
    end
    last_frame = frames(:,:,:,size(frames,4));
    
    % crop a square from center
    frames_crop = zeros(frame_h,frame_h,3,size(frames,4));
    for i = 1:size(frames_crop,4)
        frames_crop(:,:,:,i)=imcrop(frames(:,:,:,i),[offx offy (frame_h-1) (frame_h-1)]);
    end
    clear frames
    
    % scale to sz
    for i = 1:size(frames_crop,4)
        frames_small(:,:,:,fsta+i-1)=imresize(frames_crop(:,:,:,i),[sz sz]);
    end
    clear frames_crop
end

data = zeros(size(frames_small,4),sz*sz*3);
for i = 1:size(frames_small,4)
    d1 = frames_small(:,:,1,i)';
    d2 = frames_small(:,:,2,i)';
    d3 = frames_small(:,:,3,i)';
    data(i,:) = [d1(:); d2(:); d3(:)]';
end

% % watch
% for i = 1:size(frames_small,4)
%    image(frames_small(:,:,:,i)/256); 
%    pause(0.1);
% end
