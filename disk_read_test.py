import time
import threading
import numpy as np
import scipy.io

def job(k):
    path = "/mnt/ssd0/sainaa/work/121206_hdataAB_27x27x100_%d.mat" % k
    #scipy.io.loadmat(path)
    f = open(path, "rb")
    try:
        byte = "dd"
        while byte != "":
            # Do stuff with byte.
            byte = f.read(1024*1024)
    finally:
        f.close()


def task(n):
    t = []
    for k in range(n):
        t.append(threading.Thread(target = job, args = (k+1,)))
        t[k].start()

    for k in range(n):
        t[k].join()

s = time.time()
#scipy.io.loadmat("/mnt/ssd0/sainaa/work/121206_hdataAB_27x27x100_2.mat")
#scipy.io.loadmat("/home/sainaa/auto-pooling/work/121206_hdataAB_27x27x100_2.mat")
task(10)
print "time %f" % (time.time() - s)
