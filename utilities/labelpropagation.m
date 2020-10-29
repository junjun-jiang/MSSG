function [label_pre] = labelpropagation(trainlabel_nl, A)

T = zeros(size(trainlabel_nl,2),max(trainlabel_nl));
for iCTrain=1:max(trainlabel_nl) 
    ind = find(trainlabel_nl==iCTrain);
    T(ind,iCTrain)=1;
end

D = diag(sum(A));
L = D-A;
L2 = D^(-0.5)*L*D^(-0.5);

lambda = 10;
F = pinv(eye(size(L))+lambda*L)*T;%lambda in Eq. (9)
[~, label_pre] = max(F, [], 2); %simply checking which of elements is largest in each row
label_pre = label_pre';