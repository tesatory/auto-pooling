function view_pool(pool, sz, rfSize)
psz = sz - rfSize + 1;
s = psz^2;
a = reshape(pool(:,1,:),s,200);
a((1:s)+s,:) =  reshape(pool(:,2,:),s,200);
a((1:s)+s*2,:) =  reshape(pool(:,3,:),s,200);
view_data(a',psz,psz,10,20);