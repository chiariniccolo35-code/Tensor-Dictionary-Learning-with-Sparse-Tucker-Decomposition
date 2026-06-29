function [E,M1,M2,M3,r]=efficientTOMP2(A,B,C,Y,tmax,eps,itermax)
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%INPUT:
%%% A, B, C: mode dictionaries
%%% Y: input tensor
%%% tmax=s1*s2*s3 where si=sparsity density 
%%% eps: tolerance parameter
%%%itermax: maximum number of iterations
%OUTPUT:
%%% M1, M2, M3 set of indexes
%%% E: tensor s.t X(M1,M2,M3)=E will be our sparse core tensor
%%% r: vector of residual errors
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    M1=[];
    M2=[];
    M3=[];
    %[I1,I2,I3]=size(Y);
    R=tensor(Y);
    t=1;
    r(1)=1;
    while length(M1)*length(M2)*length(M3)<tmax && r(t)>eps && t<itermax
        S=ttm(R,{A',B',C'});
        S(M1,M2,M3)=0;    % in this way we will not select those indeces two or more times
        %solving argmax_{m1,m2,m3} || R x_1 A'(:,m1) x_2 B'(:,m2) x_3 C'(:,m3) ||
        % [M,ii1]=max(abs(double(S)));
        % [M,ii2]=max(M);
        % [M,ii3]=max(M);
        % m3=ii3;
        % m2=ii2(m3);
        % m1=ii1(m2);
        [~, idx] = max(abs(S(:)));
        [m1, m2, m3] = ind2sub(size(S), idx);
        M1=[M1,m1];
        M2=[M2,m2];
        M3=[M3,m3];
        D1=A(:,M1);
        D2=B(:,M2);
        D3=C(:,M3);
        %solving argmin_u || (D3 kron D2 kron D1)u -y||
        %E= ttm(tensor(Y), {pinv(D1), pinv(D2), pinv(D3)});  
        %R=Y-ttm(E,{D1,D2,D3});
        %t=t+1;
         for als_iter = 1:itermax
            % Modo 1
            S1 = double(tenmat(ttm(tensor(Y), {D2', D3'}, [2,3]), 1));
            E_mat = pinv(D1) * S1;   % |M1| × |M2|*|M3|
            E = tensor(reshape(E_mat, [size(D1,2), size(D2,2), size(D3,2)]));
            
            % Modo 2
            S2 = double(tenmat(ttm(tensor(Y), {D1', D3'}, [1,3]), 2));
            E_mat2 = pinv(D2) * S2;
            E = tensor(reshape(E_mat2', [size(D1,2), size(D2,2), size(D3,2)]));
            
            % Modo 3
            S3 = double(tenmat(ttm(tensor(Y), {D1', D2'}, [1,2]), 3));
            E_mat3 = pinv(D3) * S3;
            E = tensor(reshape(E_mat3', [size(D1,2), size(D2,2), size(D3,2)]));
         end
        R = tensor(Y) - ttm(E, {D1, D2, D3});
        t = t+1;
        r(t)=norm(R);
    end
end