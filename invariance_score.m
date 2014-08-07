function score = invariance_score(hdataA, hdataB)
N = size(hdataA,1);
r = randperm(N);

score = mean(sqrt(sum((hdataA - hdataB(r,:)).^2,2))) / ...
    mean(sqrt(sum((hdataA - hdataB).^2,2)));