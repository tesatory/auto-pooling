[W1,W2,b1,b2] = train_sparseAE_linear(dataAw,1000);
hdataA = 1 ./ (1 + exp(- bsxfun(@plus,W1 * dataAw',b1)));
hdataB = 1 ./ (1 + exp(- bsxfun(@plus,W1 * dataBw',b1)));
pool = train_pool_ae(hdataA',hdataB',100,1000,pool);