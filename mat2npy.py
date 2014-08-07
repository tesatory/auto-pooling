import numpy as np
import scipy.io

for k in range(100):
    print k
    mat = scipy.io.loadmat("/mnt/ssd0/sainaa/vimio_hdataAB_27x27x100_%d.mat" % (k+1))
    np.save("/mnt/ssd0/sainaa/vimio_hdataAB_27x27x100_%dA.npy" % (k+1), mat.get('hdataA')) 
    np.save("/mnt/ssd0/sainaa/vimio_hdataAB_27x27x100_%dB.npy" % (k+1), mat.get('hdataB')) 
