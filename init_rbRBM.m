% initialize rbRBM

% meta parameters
lrate = 0.01; % for W and bv
lrate2 = 0.1; % for bh

batchsz = 100;
drate = 0.000;
max_epoch = 2000;
CDk = 1;  % the number of gibbs sampling in contrastive divergence

% weights
bv = randn(data_dim,1) * 0.01;
bh = (randn(hid_num,1) * 0.1 + 0.7);
W = randn(hid_num,data_dim) * 0.01;

epoch = 1;
clear err;
temp = 1;
h0 = 0.1;
