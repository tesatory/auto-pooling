hdataA = bsxfun(@minus,hdataA,mean(hdataA,1));
hdataB = bsxfun(@minus,hdataB,mean(hdataB,1));
hdataA = hdataA.^2;
hdataB = hdataB.^2;
hdataA = bsxfun(@minus,hdataA,mean(hdataA,1));
hdataB = bsxfun(@minus,hdataB,mean(hdataB,1));
hdataA = bsxfun(@rdivide,hdataA,std(hdataA,1));
hdataB = bsxfun(@rdivide,hdataB,std(hdataB,1));
EC = hdataA' * hdataB / size(hdataA,1);