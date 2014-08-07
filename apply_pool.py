import sys
import time
import numpy as np
import cudamat as cm
import scipy.io

if len(sys.argv) != 5:
    print "usage: " + argv[0] + " <pool_path> <train_path> <test_path> <out_path>"
    sys.exit()

pool_path = sys.argv[1]
train_path = sys.argv[2] + "%d.npy"
test_path = sys.argv[3] + "%d.npy"
out_path = sys.argv[4]

n_train_file = 50
n_test_file = 10
filesz = 1000
trainsz = filesz * n_train_file
testsz = filesz * n_test_file
batchsz = 1000

tic = time.time()
cm.cublas_init()
pool = scipy.io.loadmat(pool_path)
W = cm.CUDAMatrix(pool.get('W'))
data_dim = W.shape[0]
pool_dim = W.shape[1]
trainXP = np.zeros((pool_dim, trainsz))
testXP = np.zeros((pool_dim, testsz))
XP_gpu = cm.empty((pool_dim, batchsz))
for n in range(n_train_file):
    print "processing train %d" % n
    XC = np.load(train_path % (n+1))
    for i in range(filesz/batchsz):
        XC_gpu = cm.CUDAMatrix(XC[:,i*batchsz:(i+1)*batchsz])
        cm.dot(W.T, XC_gpu, target=XP_gpu)
        trainXP[:,n*filesz+i*batchsz:n*filesz+(i+1)*batchsz] = XP_gpu.asarray()

for n in range(n_test_file):
    print "processing test %d" % n
    XC = np.load(test_path % (n+1))
    for i in range(filesz/batchsz):
        XC_gpu = cm.CUDAMatrix(XC[:,i*batchsz:(i+1)*batchsz])
        cm.dot(W.T, XC_gpu, target=XP_gpu)
        testXP[:,n*filesz+i*batchsz:n*filesz+(i+1)*batchsz] = XP_gpu.asarray()
cm.shutdown()

scipy.io.savemat(out_path , {"trainXC":trainXP,"testXC":testXP})    
print "time %f" % (time.time() - tic)
