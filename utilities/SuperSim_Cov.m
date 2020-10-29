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


function Sim = SuperSim_Cov(X,labels);

X = reshape(X,[],size(X,3));

for i=0:max(labels(:))    
    ind = find(labels==i);
    temp = X(ind,:);
    x_cov = covMatrix(temp');
    MeanMat(i+1,:) = x_cov(:);   
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


function x_cov = covMatrix(X)
tol = 1e-3;
tt_RD_DAT = X;
norm_tmp = sqrt(sum(tt_RD_DAT.^2));
norm_blocks_2d = tt_RD_DAT./repmat(norm_tmp,size(X,1),1);
center_pixel = mean(norm_blocks_2d,2);
cor = center_pixel'*norm_blocks_2d;
[val,sort_id] = sort(cor,'descend');
sli_id = sort_id(1:ceil(0.90*length(sort_id)));
tmp_mat = tt_RD_DAT(:,sli_id);
tmp_mat = scale_func(tmp_mat);
mean_mat = mean(tmp_mat,2);
centered_mat = tmp_mat-repmat(mean_mat,1,size(tmp_mat,2));
tmp = centered_mat*centered_mat'/((size(tmp_mat,2))-1);
x_cov = logm(tmp+tol*eye(size(tmp,1))*(trace(tmp))); 