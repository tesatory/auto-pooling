hu = [6400 12800];
MofK = [1 3 6 12 24];
Wall = cell(length(MofK),length(hu));
train_acc = zeros(length(MofK),length(hu));
test_acc = zeros(length(MofK),length(hu));
for i = 1:length(hu)
    for j = 1:length(MofK)
        W = train_multi_kmeans(patches,hu(i),200,MofK(j));
        Wall{j,i} = W;
        test_32x32
        train_acc(j,i) = train_accuracy;
        test_acc(j,i) = test_accuracy;
    end
end