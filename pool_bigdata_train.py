import sys
import time
import threading
import Queue
import numpy as np
import scipy.io
import optparse
from poolAE import *

def extract_features(X, W, b, rfsize, imsize, M, P):
    M = imsize - rfsize + 1
    N = W.shape[0]
    a = np.arange(imsize**2 * 3).reshape(imsize, imsize, 3)
    r = np.arange(N) * M**2
    Y = np.zeros((X.shape[0], N * M**2))
    c = 0
    for offx in range(M):
        for offy in range(M):
            j = a[offy:offy+rfsize,offx:offx+rfsize,:].reshape(rfsize**2 * 3)
            patches = np.dot(X[:,j]-M, P) # whiten
            Y[:,r+c] = 1 / (1 + np.exp(-np.dot(patches, W.T) - b.T))
            c += 1
    
    return Y


# move data from harddisk to main memory
def cpu_data_load_thread():      
    while data_file_queue.empty() == False:
        if options.debugmode:
            print "queue sz %d - %d - %d" % (data_file_queue.qsize(), cpu_data_queue.qsize(), gpu_data_queue.qsize())
        [pathA, pathB] = data_file_queue.get()
        A = np.load(pathA)
        B = np.load(pathB)
        cpu_data_queue.put([A, B])


# move data from memory to GPU
def gpu_data_load_thread():
    # select GPU
    cm.cuda_set_device(options.device)
    cost_buf = list()
    epoch = 1

    while True:
        s = time.time()
        if options.debugmode:
            print "queue sz %d - %d - %d" % (data_file_queue.qsize(), cpu_data_queue.qsize(), gpu_data_queue.qsize())
        [hdataA, hdataB] = cpu_data_queue.get()
        for k in range(data_file_sz / batchsz):
            hdataA_gpu = cm.CUDAMatrix(hdataA[:, k*batchsz:(k+1)*batchsz-1])
            hdataB_gpu = cm.CUDAMatrix(hdataB[:, k*batchsz:(k+1)*batchsz-1])
            #gpu_data_queue.put([hdataA_gpu, hdataB_gpu])

            # calc cost
            cost = pool.train_step(hdataA_gpu, hdataB_gpu)
            cost_buf.append(cost.asarray()[0,0])
            if len(cost_buf) == data_sz / batchsz:
                print "epoch %d cost %f" % (epoch, (sum(cost_buf)/len(cost_buf)))
                cost_list.append(sum(cost_buf)/len(cost_buf))
                cost_buf = []
                if epoch == 5:
                    pool.moment = 0.9
                if epoch % 10 == 0: 
                    scipy.io.savemat(options.out_path + ("_ep%d.mat" % epoch), {'W': pool.Wgpu.asarray(), 'cost': cost_list})
                epoch += 1

        cpu_data_queue.task_done()
        if options.debugmode:
            print "one batch time %f" % (time.time() - s)


#def train_step_thread():
#    cost_buf = []
#    ep = 1
#    while True:
#        print "queue sz %d - %d - %d" % (data_file_queue.qsize(), cpu_data_queue.qsize(), gpu_data_queue.qsize())
#        [hdataA, hdataB] = gpu_data_queue.get()
#        cost =  pool.train_step(hdataA, hdataB)
#        cost_buf.append(cost.asarray()[0,0])
#        if len(cost_buf) == data_sz / batchsz:
#            print "epoch %d cost %f" % (ep, (sum(cost_buf)/len(cost_buf)))
#            cost_buf = []
#            ep += 1
#        gpu_data_queue.task_done()


parser = optparse.OptionParser()
parser.add_option("-n", "--datadim", dest="data_dim", type="int", \
                      help="data dimension", default=27*27*100)
parser.add_option("-m", "--pooldim", dest="pool_dim", type="int", \
                      help="pool dimension", default=400)
parser.add_option("-e", "--epoch", dest="max_epoch", type="int", default=1)
parser.add_option("-r", "--lrate", dest="lrate", type="float", \
                      default=0.001, help="learning rate")
parser.add_option("-l", "--lambda", dest="lam", type="float", default=8)
parser.add_option("-i", "--in", dest="in_path", \
                      default="/mnt/ssd0/sainaa/vimio_hdataAB_27x27x100_")
parser.add_option("-o", "--out", dest="out_path", default="/tmp/pool_big_out")
parser.add_option("-f", "--pool", dest="pool_path")
parser.add_option("-b", "--batchsz", dest="batchsz", type="int", default=100)
parser.add_option("-v", "--verbose", dest="debugmode", action="store_true")
parser.add_option("-d", "--device", dest="device", type="int", default=0)

(options, args) = parser.parse_args()
if options.out_path == None:
    print "no output"
    sys.exit()


# select GPU
cm.cuda_set_device(options.device)

# parameters
data_dim = options.data_dim
pool_dim = options.pool_dim
max_epoch = options.max_epoch
batchsz = options.batchsz
data_file_count = 100
data_file_sz = 1000
data_path_patternA = options.in_path + "%dA.npy"
data_path_patternB = options.in_path + "%dB.npy"

# init poolAE
pool = poolAE(data_dim, pool_dim)
pool.lam = options.lam
pool.lrate = options.lrate
if options.pool_path != None:
    pool_old = scipy.io.loadmat(options.pool_path)
    pool.W = pool_old.get('W')
pool.train_init()

# system parameters
#gpu_max_bytes = 500 * 1024 * 1024 # 500MB
cpu_max_bytes = 3 * 1024 * 1024 * 1024 # 5GB
n_cpu_loader = 4

# make a job queue
train_start = time.time()
cost_list = list()
data_sz = data_file_count * data_file_sz
data_file_queue = Queue.Queue()
for epoch in range(max_epoch):
    for nfile in range(data_file_count):
        data_file_queue.put([data_path_patternA % (nfile+1), \
                                 data_path_patternB % (nfile+1)])

# create workers for data loading
cpu_batch_bytes = 2 * data_dim * data_file_sz * 4
cpu_queue_max = cpu_max_bytes / cpu_batch_bytes
print "cpu queue max %d" % cpu_queue_max
cpu_data_queue = Queue.Queue(cpu_queue_max)

# create workers for training on GPU
#gpu_batch_bytes = 2 * data_dim * batchsz * 4
#gpu_queue_max = gpu_max_bytes / gpu_batch_bytes
gpu_queue_max = 1 # no concurrent access device memory
print "gpu queue max %d" % gpu_queue_max
gpu_data_queue = Queue.Queue(gpu_queue_max)

cpu_loader_list = []
for i in range(n_cpu_loader):
    cpu_loader = threading.Thread(target = cpu_data_load_thread)
    cpu_loader.setDaemon(True)
    cpu_loader.start()
    cpu_loader_list.append(cpu_loader)

gpu_loader = threading.Thread(target = gpu_data_load_thread)
gpu_loader.setDaemon(True)
gpu_loader.start()

#trainer = threading.Thread(target = train_step_thread)
#trainer.setDaemon(True)
#trainer.start()

for i in range(n_cpu_loader):
    cpu_loader_list[i].join() # all data must be loaded
print "cpu loader finished"
cpu_data_queue.join() # 
print "gpu loader finished"
#gpu_data_queue.join() #
#print "trainer finished"

pool.train_finalize()
scipy.io.savemat(options.out_path + ".mat", {'W': pool.W, 'cost': cost_list})
print "train time %f" % (time.time() - train_start)


