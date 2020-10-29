function Lnew = sc_ml(A,K,lambda)
%% the SC-ML algorithm
% A: multi-layer graph
% K: target number of clusters
% lambda: regularization parameter

% paramter setting
N = size(A,1); % number of vertices in each layer
M = size(A,3); % number of layers

% compute the low dimensional embeddings
L = zeros(N,N,M);
V = zeros(N,K,M);
D = zeros(K,K,M);

for m = 1:M
    L(:,:,m) = eye(N)-normadj(A(:,:,m));
    % Normalized Laplacian can be computed by substracting normalized 
    % adjacency matrix from the identity matrix;
    % However, this would work only in the case when there is no isolated
    % nodes in the graph (which is the case we assume)
    [V(:,:,m),D(:,:,m)] = eigs(L(:,:,m),K,'sa');
end

% compute modified Laplacian
Lmod = zeros(N,N);
for i = 1:M
    Lmod = Lmod + V(:,:,i)*V(:,:,i)';
end

% compute the representative subspace
Lnew = sum(L,3) - lambda*Lmod;
% [Vnew,Dnew] = eigs(Lnew,K,'sa');
% 
% % clustering implementation
% Vk = rnorm(Vnew);
% [idx,ctrs] = kmeans(Vk,K,'emptyaction','singleton','replicates',500);



function N = rnorm(M)
%% normalize the rows of matrix M
N = M;

for i = 1:size(M,1)
    if norm(M(i,:)) ~= 0
        N(i,:) = M(i,:)./norm(M(i,:));
    end
end



function A = normadj(A)
%% compute the normalized adjacency matrix

% This code is extracted from the file sgwt_laplacian.m, which is part of
% the SGWT toolbox (Spectral Graph Wavelet Transform toolbox)
% Copyright (C) 2010, David K. Hammond. 
%
% The SGWT toolbox is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% The SGWT toolbox is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with the SGWT toolbox.  If not, see <http://www.gnu.org/licenses/>.

N=size(A,1);
degrees=vec(full(sum(A)));
% to deal with loops, must extract diagonal part of A
diagw=diag(A);

% w will consist of non-diagonal entries only
[ni2,nj2,w2]=find(A);
ndind=find(ni2~=nj2); % as assured here
ni=ni2(ndind);
nj=nj2(ndind);
w=w2(ndind);

di=vec(1:N); % diagonal indices  

% normalized laplacian D^(-1/2)*(D-A)*D^(-1/2)
% diagonal entries
dL=(diagw./degrees); % will produce NaN for degrees==0 locations
dL(degrees==0)=0;% which will be fixed here
% nondiagonal entries
ndL=w./vec( sqrt(degrees(ni).*degrees(nj)) );
L=sparse([ni;di],[nj;di],[ndL;dL],N,N);
A=full(L);



function r=vec(x)
r=x(:);