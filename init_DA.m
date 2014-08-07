% initialize Denoising AutoEncoder

% meta parameters
lrate = 0.01;
batchsz = 100;
corruption = 1.5;
max_epoch = 2000;

% weights
bv = zeros(data_dim,1);
bh = zeros(hid_num,1);
l = 4 * sqrt(6 / (hid_num + data_dim));
W = rand(hid_num,data_dim) * 2 * l - l;
W_prime = W';

clear cost
c = 1;
epoch = 1;

data_w = sqrt(data_dim);
data_h = sqrt(data_dim);

binary = false;
weight_tied = false;