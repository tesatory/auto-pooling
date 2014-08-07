function p = trans_prob_direct_calc(hdataA,hdataB)
p_xy = hdataA' * hdataA / 10000;
p_x = sum(hdataA)' / 10000;
p_yGx = bsxfun(@rdivide, p_xy, p_x);
p_xz = hdataA' * hdataB / 10000;
p_zGx = bsxfun(@rdivide, p_xz, p_x);
p_yGx_inv = p_yGx^-1;
p = p_yGx_inv * p_zGx;