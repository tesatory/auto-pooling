import time
import numpy as np
import cudamat as cm

class poolAE:
    lrate = 0.01
    batchsz = 100
    bigbatchsz = 10000
    moment = 0
    lam = 2
    max_moment = 0.9
    
    def __init__(self, _data_dim, _pool_dim):
        self.data_dim = _data_dim
        self.pool_dim = _pool_dim
        self.W = np.zeros((self.data_dim, self.pool_dim))
        for i in range(self.pool_dim):
            self.W[np.random.randint(0, self.data_dim), i] = 1

    def calc_cost(self, X1, X2):
        P1 = cm.dot(self.Wgpu.T, X1)
        P2 = cm.dot(self.Wgpu.T, X2)
        Y1 = cm.dot(self.Wgpu, P1)
        Y2 = cm.dot(self.Wgpu, P1)

        Y1.subtract(X1)
        Y2.subtract(X2)
    
        PD = cm.empty(P1.shape)
        XD = cm.empty(X1.shape)
        P1.subtract(P2, target = PD)
        X1.subtract(X2, target = XD)
    
        grad = cm.dot(XD, PD.T)
        grad.mult(self.lam)
        grad.add(cm.dot(Y1, P1.T))
        grad.add(cm.dot(Y2, P2.T))
        grad.add(cm.dot(X1, cm.dot(Y1.T, self.Wgpu)))
        grad.add(cm.dot(X2, cm.dot(Y2.T, self.Wgpu)))
        grad.divide(X1.shape[1])

        PD.mult(PD)
        Y1.mult(Y1)
        Y2.mult(Y2)
        cost = PD.sum(axis = 0)
        cost.mult(self.lam)
        cost.add(Y1.sum(axis = 0))
        cost.add(Y2.sum(axis = 0))
        cost = cost.sum(axis = 1)
        cost.divide(X1.shape[1] * 2)

        return [cost, grad]

    def train_step(self, X1, X2):
        [cost, grad] = self.calc_cost(X1, X2)
    
        grad.mult(self.lrate)
        self.speed.mult(self.moment)
        self.speed.subtract(grad)
        self.Wgpu.add(self.speed)
        Wmask = cm.empty(self.W.shape)
        self.Wgpu.greater_than(0, target = Wmask)
        self.Wgpu.mult(Wmask)
        
        return cost
        
    def train_init(self):
        # init cudamat
        cm.cublas_init()
        cm.CUDAMatrix.init_random(1)

        self.Wgpu = cm.CUDAMatrix(self.W)
        self.speed = cm.empty(self.W.shape)
        self.speed.assign(0)


    def train_finalize(self):
        self.Wgpu.copy_to_host()
        self.W = self.Wgpu.numpy_array
        print "CUDA try shutdown"
        cm.cublas_shutdown()
       

    def train(self, X1, X2, max_epoch):
        self.train_init()

        datasz = X1.shape[1]
 
        X1gpu = cm.CUDAMatrix(X1)
        X2gpu = cm.CUDAMatrix(X2)
        X1sub = cm.empty((self.data_dim, self.batchsz))
        X2sub = cm.empty((self.data_dim, self.batchsz))

        tic = time.time()
        for epoch in range(max_epoch):
            if epoch == 5:
                self.moment = 0.9

            for n in range(datasz / self.batchsz):
                X1gpu.get_col_slice(self.batchsz * n, self.batchsz * (n+1), target = X1sub)
                X2gpu.get_col_slice(self.batchsz * n, self.batchsz * (n+1), target = X2sub)
                cost = self.train_step(X1sub, X2sub)
                
            print "epoch %d cost %f" % (epoch, cost.asarray()[0,0])

        toc = time.time()
        print "time %s" % (toc - tic)
        self.train_finalize()

#p = poolAE(1000,400)
#p.lrate = 0.001
#p.bigbatchsz = 10000
#X1 = np.random.rand(1000,1000)        
#X2 = np.random.rand(1000,1000)
#p.train(X1,X2,10)
