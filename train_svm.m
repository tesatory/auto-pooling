function theta = train_svm(trainXC, trainY, C)
  
  numClasses = max(trainY);
  %w0 = zeros(size(trainXC,2)*(numClasses-1), 1);
  w0 = zeros(size(trainXC,2)*numClasses, 1);
  w = minFunc(@my_l2svmloss, w0, struct('MaxIter', 1000, 'MaxFunEvals', 1000), ...
              trainXC, trainY, numClasses, C);
          
% options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
%                           % function. Generally, for minFunc to work, you
%                           % need a function pointer with two outputs: the
%                           % function value and the gradient. In our problem,
%                           % sparseAutoencoderCost.m satisfies this.
% options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run 
% options.display = 'on';
% 
% w = minFunc( @(p) my_l2svmloss(p, ...
%                                    trainXC, trainY, ...
%                                    numClasses, C), ...
%                               w0, options);
                          
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
