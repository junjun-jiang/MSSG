function [Aout] = supSimALLMultiscale(indian_pines_corrected,indian_pines_gt,train,randpp,labels,st)
C = (size(labels,3)-1)/3;
l = var(1:C);
weight = [exp(-abs(-C:0).^2/l) exp(-abs(1:2*C).^2/l)];
weight = weight./sum(weight);
Aout = zeros(st);

% C = floor(size(labels,3)/2);
% if C==0
%     weight = 1;
% else
%     if C==1
%         l=1;
%     else
%         l = var(1:C);
%     end
%     weight = [exp(-abs(-C:0).^2/l) exp(-abs(1:C).^2/(C*l))];
%     weight = weight./sum(weight);   
% end
% Aout = zeros(st);

for iseg = 1:size(labels,3)
    Aout = Aout+weight(iseg).*supSimALL(indian_pines_corrected,indian_pines_gt,train,randpp,labels(:,:,iseg));  
end