function [label_pre] = randlabelpropagation(trainlabel_nl, A, gamma,para_alpha)

T = zeros(size(trainlabel_nl,2),max(trainlabel_nl));
for iCTrain=1:max(trainlabel_nl) 
    ind = find(trainlabel_nl==iCTrain);
    T(ind,iCTrain)=1;
end

% fea = [DataTrain];
% options = [];
% options.NeighborMode = 'KNN';
% options.k = 25;
% options.WeightMode = 'HeatKernel';
% options.t = 10;
% A2 = constructW(fea,options);A2 = full(A2);
% A2 = A2./repmat(sum(A2,2),1,size(A2,1));
% A = 0.7*A+0.3*A2;

% load('SimMatrix.mat','A');

% save(['SimMatrix-', num2str(iter),'.mat'],'A');
% alpha_ls = 1-exp(-tan(1-sigma*3.14/2));
% alpha_us = 1-alpha_ls;

S = round(0.0*size(trainlabel_nl,2));

% if gamma<=0.15
%     AA = inv(eye(size(A,1)) - 0.97 * A);
% else
%     AA = inv(eye(size(A,1)) - 0.90 * A);
% end

AA = inv(eye(size(A,1)) - para_alpha * A);


for iter = 1:1
    Tr = T;
    temprand = randperm(size(Tr,1));
    Tr(temprand(1:S),:)=0;
    
    Tr = [Tr;zeros(size(AA,1)-size(trainlabel_nl,2),size(Tr,2))];%拼上测试样本的标签矩阵（全0）

%     alpha_l = alpha_ls(ialpha_l);
%     alpha_u = alpha_us(ialpha_u);
%     is_normalize = 1;            
%     F = GeneralSSL(A, Tr, alpha_l, alpha_u, is_normalize);
%     [val,label(iter,:)] = max(F');

    F = AA * Tr; % classification function F = (I - \alpha S)^{-1}Y
    [~, label_temp] = max(F, [], 2); %simply checking which of elements is largest in each row
    label(iter,:) = label_temp(1:size(trainlabel_nl,2));
end
[label_pre] = label_fusion(label');

% for iter = 1:100
%     Tr = T;
%     temprand = randperm(size(Tr,1));
%     Tr(temprand(temprand(1:S)),:)=0;
%     
%     Tr = [Tr;zeros(size(AA,1)-size(trainlabel_nl,2),9)];
% 
%     alpha_l = 0.9;
%     alpha_u = 0.8;
%     is_normalize = 1;            
%     F = GeneralSSL(A, Tr, alpha_l, alpha_u, is_normalize);
%     [val,label(iter,:)] = max(F');
% 
% end 
% [weights, label_pre] = label_fusion(label');


% T(ind,:)=0;
% for iter = 1:300
%     Tr = T;
%     temprand = randperm(size(Tr,1));
%     Tr(temprand(1:S),:)=0;
% 
% %     alpha_l = alpha_ls(ialpha_l);
% %     alpha_u = alpha_us(ialpha_u);
% %     is_normalize = 1;            
% %     F = GeneralSSL(A, Tr, alpha_l, alpha_u, is_normalize);
% %     [val,label(iter,:)] = max(F');
% 
%     F = AA * Tr; % classification function F = (I - \alpha S)^{-1}Y
%     [~, label(iter,:)] = max(F, [], 2); %simply checking which of elements is largest in each row
% end
% [weights, label_pre] = label_fusion(label');


label_pre = label_pre';






