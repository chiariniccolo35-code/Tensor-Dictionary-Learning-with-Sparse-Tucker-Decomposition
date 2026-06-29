function [A,B,C,X_cb,A0,B0,itrs,resto]=GradTensor(Y,X,gamma,eps,A0,B0,C0,itrs_max)
    Y_1=unfold(Y,1);
    Y_2=unfold(Y,2);
    Y_3=unfold(Y,3);
    I=size(Y);
    M=size(X);
    M1=M(1);
    M2=M(2);
    M3=M(3);
    I1=I(1);
    I2=I(2);
    I3=I(3);
    X_cb=ttm(X,{C0,B0},[3,2]);
    X_cb=unfold(tensor(X_cb),1);
    %size(X_cb)
    %X_cb=double(reshape(X_cb,[M1,I2*I3]));
    A=A0;
    itrs=0;
    res=1;
    %resto=res;
    %grad_1=(Y_1-A*X_cb)*pinv(X_cb);
    Pinv_cb=pinv(X_cb);
    resto(1)=norm(Y-ttm(X,{A0,B0,C0}));
    %add_cb=Y_1*Pinv_cb;
    while res>eps && itrs<itrs_max
        A_old=A;
        grad_1=(Y_1-A*X_cb)*Pinv_cb;
        A=A+ gamma *grad_1;
        res=norm(A-A_old);
        itrs=itrs+1;
        %resto(itrs)=norm(Y-ttm(tensor(X),{A,B0,C0}));
    end
    if res<eps
        flag(1)=1;
    else
        flag(1)=0;
    end
    resto(2)=norm(Y-ttm(X,{A,B0,C0}));
    itrsA=itrs;
    itrs=0;
    res=1;
    X_ca=ttm(X,{C0,A},[3,1]);
    X_ca=unfold(tensor(X_ca),2);
    %X_ca=double(reshape(X_ca,[M2,I1*I3]));
    Pinv_ca=pinv(X_ca);
    B=B0;
    %add_ca=Y_2*Pinv_ca;
    while res>eps && itrs<itrs_max
        B_old=B;
        grad_2=(Y_2-B*X_ca)*Pinv_ca;
        B=B+ gamma *grad_2;
        res=norm(B-B_old);
        itrs=itrs+1;
       % resto(itrs+itrsA)=norm(Y-ttm(tensor(X),{A,B,C0}));
    end
     if res<eps
        flag(2)=1;
    else
        flag(2)=0;
    end
    resto(3)=norm(Y-ttm(X,{A,B,C0}));
    itrsB=itrs;
    itrs=0;
    res=1;
    X_ba=ttm(X,{B,A},[2,1]);
    X_ba=unfold(tensor(X_ba),3);
    %X_ba=double(reshape(X_ba,[M3,I1*I2]));
    Pinv_ba=pinv(X_ba);
    % add_ba=Y_3*Pinv_ba;
    C=C0;
    while res>eps && itrs<itrs_max
        C_old=C;
        grad_3=(Y_3 -C*X_ba)*Pinv_ba;
        C=C+ gamma *grad_3;
        res=norm(C-C_old);
        itrs=itrs+1;
        %resto(itrs+itrsA+itrsB)=norm(Y-ttm(tensor(X),{A,B,C0}));

    end
     if res<eps
        flag(3)=1;
    else
        flag(3)=0;
    end
    resto(4)=norm(Y-ttm(X,{A,B,C}));
end
