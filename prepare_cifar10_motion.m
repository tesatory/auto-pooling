% data = (data - min(data(:))) / (max(data(:)) - min(data(:)));
dataA = zeros(100000,16*16*3);
dataB = zeros(size(dataA));

for i=1:size(dataA,1)
    if mod(i,1000) == 0
        fprintf('preparing %d / %d\n',i,size(dataA,1));
    end
    a = reshape(trainX(mod(i-1,50000)+1,:), 32, 32, 3);
    b = a(9:24, 9:24,:);
    dataA(i,:) = b(:)';
    while true
        dx = round(randn*1.5);
        dy = round(randn*1.5);
%         dx = randi([-1 1]);
%         dy = randi([-1 1]);
%         dx = ceil(exprnd(1));
%         dy = ceil(exprnd(1));
%         if rand < 0.5
%             dy = - dy;
%         end
        if 9+dx > 0 && 9+dy>0 && 24+dx <= 32 && 24+dy <= 32
            break
        end
    end
%    a = imrotate(a, round(randn * 5), 'bilinear','crop');
    b = a((9:24)+dx, (9:24)+dy,:);
    dataB(i,:) = b(:)';
end
% 
% trans = zeros(121,121);
% c = 1;
% for y = 1:11
%     for x = 1:11
%         a = zeros(11,11);
% %         for r = -2:2
% %             if 0 < y+r && y+r < 12 && 12 > x-r && x-r > 0
% %                 a(y+r,x-r) = 1;
% %             end
% %         end
%         a(max(y-2,1):min(y+2,11),max(x-1,1):min(x+1,11)) = 1;
%         trans(c,:) = a(:)';
%         c = c + 1;
%     end
% end
