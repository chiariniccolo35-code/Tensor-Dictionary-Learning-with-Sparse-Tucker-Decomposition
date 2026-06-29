function [E,E2,M1,M2,M3,r,r2]=efficientTOMP3(A,B,C,A2,B2,C2,Y,tmax,eps,itermax)
    M1=[];
    M2=[];
    M3=[];
    M12=[];
    M22=[];
    M32=[];
    [I1,J1]=size(A);
    [I2,J2]=size(B);
    [I3,J3]=size(C);
    R=tensor(Y);
    X=sptensor([],[],[J1,J2,J3]);
    y=reshape(Y,[I3,I2*I1]);
    t=1;
    r(1)=1;
    while length(M1)*length(M2)*length(M3)<tmax && r(t)>eps && t<itermax
        S=ttm(R,{A',B',C'});
        S(M1,M2,M3)=0;
        [M,ii1]=max(abs(double(S)));
        [M,ii2]=max(M);
        [M,ii3]=max(M);
        m3=ii3;
        m2=ii2(m3);
        m1=ii1(m2);
        M1=[M1,m1];
        M2=[M2,m2];
        M3=[M3,m3];
        D1=A(:,M1);
        D2=B(:,M2);
        D3=C(:,M3);

        S2=ttm(R,{A2',B2',C2'});
        S2(M12,M22,M32)=0;
        [M,ii1]=max(abs(double(S2)));
        [M,ii2]=max(M);
        [M,ii3]=max(M);
        m3=ii3;
        m2=ii2(m3);
        m1=ii1(m2);
        M12=[M12,m1];
        M22=[M22,m2];
        M32=[M32,m3];
        D12=A2(:,M12);
        D22=B2(:,M22);
        D32=C2(:,M32);
        %res=y/D3';
        % size(res)
        % size(D3)
        %res=reshape(res,[I1*length(M3),I2]);
        % size(res)
        % size(D2)
        % res=res/D2';
        % size(res)
        % res=reshape(res,[I1,length(M2)^2]);
        % e=D1\(res);
        res=D3\y;
        e=res/kron(D2,D1)';
        E=reshape(e,length(M1),length(M2),length(M3));
        E=tensor(E,[length(M1),length(M2),length(M3)]);
        %E=tensor(E,[length(M1),length(M2),length(M3)]);
        E2= ttm(tensor(Y), {pinv(D1), pinv(D2), pinv(D3)});
        R=Y-ttm(E,{D1,D2,D3});
        t=t+1;
        r(t)=norm(R);
        r2(t)=norm(E-E2);
    end
end