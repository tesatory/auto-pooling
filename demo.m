prepare_cifar10_motion
load centroids200
hdataA = extract_all(dataA, centroids, rfSize, [16 16 3], M, P);
hdataB = extract_all(dataB, centroids, rfSize, [16 16 3], M, P);
k = 7;
hdataA2 = hdataA(:,121*k-120:121*k);
hdataB2 = hdataB(:,121*k-120:121*k);