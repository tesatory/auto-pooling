function [dataA, dataB,dataC] = prepare_rand_motion_data(data,N,scale_std,rot_std,shift_std)
dataA = zeros(N,3072);
dataB = zeros(N,3072);
dataC = zeros(N,3072);

n = 1;
skip = 0;
while n <= N
    try
        if mod(n,1000) == 0
            fprintf('%d / %d (%d)\n',n,N,skip);
        end
        m = mod(n-1,size(data,1)) + 1;
        I = reshape(data(m,:),32,32,3);
        scaleA = randn * scale_std;
        scaleB = randn * scale_std;
        Asz = round(scaleA * 32)+32;
        Bsz = round(scaleB * 32)+32;
        A = imresize(I,[Asz Asz],'bilinear');
        B = imresize(I,[Bsz Bsz],'bilinear');
        A = imrotate(A,randn*rot_std,'bilinear','crop');
        B = imrotate(B,randn*rot_std,'bilinear','crop');
        ca = round((Asz-32)/2)+1:round((Asz-32)/2)+32;
        cb = round((Bsz-32)/2)+1:round((Bsz-32)/2)+32;
        A = A(ca + round(randn * shift_std),ca + round(randn * shift_std),:);
        B = B(cb + round(randn * shift_std),cb + round(randn * shift_std),:);
        dataA(n,:) = A(:)';
        dataB(n,:) = B(:)';
        dataC(n,:) = I(:)';
        n = n + 1;
    catch err
        skip = skip + 1;
    end
end
fprintf('skipped %d\n',skip);