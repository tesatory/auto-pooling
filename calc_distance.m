function s = calc_distance(W, data1, data2) 
s = W * (data1 - data2)';
s = mean(s.^2,2);