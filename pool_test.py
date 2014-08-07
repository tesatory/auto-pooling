import scipy.io
from poolAE import *
import pdb

ae = scipy.io.loadmat('work/121129_sparse_cifar16g_1000hu_c5t2_epoch3k.mat')
d =  scipy.io.loadmat('work/121129_video_patches_16x16gray_50k_nostill.mat')

W1 = ae.get('W1')
b1 = ae.get('b1')
M = ae.get('M')
P = ae.get('P')

dataA = d.get('dataA')
dataB = d.get('dataB')
dataAw = np.dot(dataA - M, P)
dataBw = np.dot(dataB - M, P)

hdataA = 1 / (1 + np.exp(- np.dot(dataAw, W1.T) - b1.T))
hdataB = 1 / (1 + np.exp(- np.dot(dataBw, W1.T) - b1.T))

pdb.set_trace()

pool = poolAE(1000,400)
pool.lam = 2
pool.train(hdataA, hdataB, 100)

scipy.io.savemat('work/121206_pool_py.mat', {'W': pool.W})
