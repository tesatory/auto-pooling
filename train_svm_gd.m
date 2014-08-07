function theta = train_svm_gd(trainXC, trainY, testXC,testY, C, maxIter,lrate)
  
  numClasses = max(trainY);
  %w0 = zeros(size(trainXC,2)*(numClasses-1), 1);
  w = zeros(size(trainXC,2)*numClasses, 1);
  loss = zeros(maxIter,1);
  train_acc_log = [];
  test_acc_log = [];
  mom = 0;
  vw = zeros(size(w));
  for n = 1:maxIter
      if n == 100
          mom = 0.9;
      end
      [l,g] = my_l2svmloss(w, trainXC, trainY, numClasses, C);
      dw = - g;
      vw = mom*vw + dw;
      w = w + lrate * vw;
      loss(n,1) = l;
      if mod(n,1) == 0
          fprintf('iter %d loss %f\n',n,l);
          subplot(2,1,1)
          plot(loss);
          drawnow;
      end
      if mod(n,10) == 0
          theta = reshape(w, size(trainXC,2), numClasses);
          [val,labels] = max(trainXC*theta, [], 2);
          train_acc = 100 * (1 - sum(labels ~= trainY) / length(trainY));
          fprintf('Train accuracy %f%%\n', train_acc);
          [val,labels] = max(testXC*theta, [], 2);
          test_acc = 100 * (1 - sum(labels ~= testY) / length(testY));
          fprintf('Test accuracy %f%%\n', test_acc);
          
          train_acc_log(length(train_acc_log)+1) = train_acc;
          test_acc_log(length(test_acc_log)+1) = test_acc;
          subplot(2,1,2)
          plot(train_acc_log)
          hold on
          plot(test_acc_log)
          hold off
      end
  end
                          
  theta = reshape(w, size(trainXC,2), numClasses);
  
% 1-vs-all L2-svm loss function;  similar to LibLinear.
function [loss, g] = my_l2svmloss(w, X, y, K, C)
  [M,N] = size(X);
  theta = reshape(w, N,K);
  Y = bsxfun(@(y,ypos) 2*(y==ypos)-1, y, 1:K);

  margin = max(0, 1 - Y .* (X*theta));
  loss = (0.5 * sum(theta.^2)) + C*mean(margin.^2);
  loss = sum(loss);  
  g = theta - 2*C/M * (X' * (margin .* Y));
  g = g(:);

  %[v,i] = max(X*theta,[],2);
  %sum(i ~= y) / length(y)
