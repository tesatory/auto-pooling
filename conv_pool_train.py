import numpy as np
import time

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

X = np.random.randn(1000,32*32*3)
W = np.random.randn(100,6*6*3)
b = np.random.randn(100,1)
M = np.random.randn(1,6*6*3)
P = np.random.randn(6*6*3,6*6*3)

tic = time.time()
Y = extract_features(X,W,b,6,32,M,P)
print str(time.time() - tic)
print Y.shape
