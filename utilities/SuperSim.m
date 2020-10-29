% function neighborhood = SuperSim(X,labels);
% X = reshape(X,[],size(X,3));
% for i=0:max(labels(:)) 
%     for j=0:max(labels(:)) 
%         ind_i = find(labels==i);
%         ind_j = find(labels==j);
%         temp_i = X(ind_i,:);temp_i = fea_norm(temp_i')';
%         temp_j = X(ind_j,:);temp_j = fea_norm(temp_j')';
%         temp_ij = temp_i*temp_j';
%         Sim(1+i,1+j) = mean(mean(temp_ij));
%     end
% end
% distance = 1-Sim;
% [~,index] = sort(distance);
% neighborhood = index;


function Sim = SuperSim(X,labels);

X = reshape(X,[],size(X,3));

for i=0:max(labels(:))    
    ind = find(labels==i);
    temp = X(ind,:);
    MeanMat(i+1,:) = mean(temp);   
end

X = MeanMat';
K = size(X,2)-1;
[Sim] = lle(X,K);

function [neighborhood] = lle(X,K)

[D,N] = size(X);

X2 = sum(X.^2,1);
distance = repmat(X2,N,1)+repmat(X2',1,N)-2*X'*X;

[~,index] = sort(distance);
neighborhood = index(1:(1+K),:);