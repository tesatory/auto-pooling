function task_121201_matlab_gpu_test()
X1 = rand(10000,10000);
X2 = rand(10000,10000);
train_pool_ae(X1,X2,400,1,10,true,true);
