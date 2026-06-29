function [E,M1,M2,M3,M4,r]=efficientTOMP4d(A,B,C,D,Y,tmax,eps,itermax)

    M1=[];
    M2=[];
    M3=[];
    M4=[];
    %[I1,I2,I3]=size(Y);
    R=tensor(Y);
    t=1;
    r(1)=1;
    while length(M1)*length(M2)*length(M3)*length(M4)<tmax && r(t)>eps && t<itermax
        S=ttm(R,{A',B',C',D'});
        S(M1,M2,M3,M4)=0;    % in this way we will not select those indeces two or more times
        %solving argmax_{m1,m2,m3} || R x_1 A'(:,m1) x_2 B'(:,m2) x_3 C'(:,m3) ||
        % [M,ii1]=max(abs(double(S)));
        % [M,ii2]=max(M);
        % [M,ii3]=max(M);
        % m3=ii3;
        % m2=ii2(m3);
        % m1=ii1(m2);
        [~, idx] = max(abs(S(:)));
        [m1, m2, m3,m4] = ind2sub(size(S), idx);
        M1=[M1,m1];
        M2=[M2,m2];
        M3=[M3,m3];
        M4=[M4,m4];
        D1=A(:,M1);
        D2=B(:,M2);
        D3=C(:,M3);
        D4=D(:,M4);
        %solving argmin_u || (D3 kron D2 kron D1)u -y||
        E= ttm(tensor(Y), {pinv(D1), pinv(D2), pinv(D3),pinv(D4)});  
        R=Y-ttm(E,{D1,D2,D3,D4});
        t=t+1;
        r(t)=norm(R);
    end
end