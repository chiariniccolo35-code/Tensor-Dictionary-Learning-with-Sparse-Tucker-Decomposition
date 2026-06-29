
function [E,M1,M2,M3,r]=TOMP(A,B,C,Y,tmax,eps,itermax)
M1=[];
M2=[];
M3=[];
J1=size(A,2);
J2=size(B,2);
J3=size(C,2);
R=tensor(Y);
X=sptensor([],[],[J1,J2,J3]);
y=vec(tensor(Y));
t=0;
while length(M1)*length(M2)*length(M3)<tmax && norm(R)>eps && t<itermax
    S=ttm(R,{A',B',C'});
    S(M1,M2,M3)=0;
    %[M,ii1]=max(abs(double(S)));
    %[M,ii2]=max(M);
    %[M,ii3]=max(M);
    [~, idx] = max(abs(double(S(:))));  % linearizza e trova indice globale

% Converti indice lineare in multiindice
    [ii1, ii2, ii3] = ind2sub(size(S_abs), idx);
    m3=ii3;
    m2=ii2(m3);
    m1=ii1(m2);
    M1=[M1,m1];
    M2=[M2,m2];
    M3=[M3,m3];
    D1=A(:,M1);
    D2=B(:,M2);
    D3=C(:,M3);
    e=kron(D3,kron(D2,D1))\y;
    E=reshape(e,length(M1),length(M2),length(M3));
    E=tensor(E,[length(M1),length(M2),length(M3)]);
    R=Y-ttm(E,{D1,D2,D3});
    t=t+1;
    r(t)=norm(R)/norm(tensor(Y));
end
end