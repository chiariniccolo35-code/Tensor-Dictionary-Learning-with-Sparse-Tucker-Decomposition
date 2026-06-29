function [A,B,C,flag,resto]=GradTensorEff(Y,X,gamma,eps,A0,B0,C0,itrs_max,decay)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Perform gradient descent on dictionaries A0, B0, C0 withh stepsize
%%%%% gamma and tolerance eps. 
%%%%% INPUT: Y input tensor
%%%%%        X input core tensor
%%%%%        A0, B0, C0 original dictionaries to be optimized
%%%%%        gamma step size
%%%%%        eps tolerance: gradient descent stops whenever 
%%%%%        itrs_max number of max iterations for GD
%%%%%        decay: after 8/10 *itrs_max iterations stepsize becomes
%%%%%        gamma*decay
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Y_1=unfold(Y,1);
    Y_2=unfold(Y,2);
    Y_3=unfold(Y,3);

    %Gradient descent for dictionary A
    X_cb=ttm(X,{C0,B0},[3,2]);
    X_cb=unfold(tensor(X_cb),1);
    A=A0;
    itrs=0;
    p=1;
    res=0;
    Pinv_cb=pinv(X_cb);
    resto(1)=norm(Y-ttm(X,{A0,B0,C0}));
    Y_term=Y_1*Pinv_cb;
    X_term=X_cb*Pinv_cb;
    gamma_or=gamma;
    bool=1;
     while p>eps && itrs<itrs_max
         A_old=A;
         grad_1=(Y_term-A*X_term);
         A=A+ gamma *grad_1;
        % res=norm(A-A_old);
         itrs=itrs+1;
         res_old=res;
         res=norm(Y-ttm(tensor(X),{A,B0,C0}));
         p=res-res_old;
         if p<eps || (itrs>(8/10 *itrs_max) && bool==1)
             gamma=decay*gamma;
             A_old=A;
             grad_1=(Y_term-A*X_term);
             A=A+ gamma *grad_1;
             res_old=res;
             res=norm(Y-ttm(tensor(X),{A,B0,C0}));
             p=res-res_old;
             itrs=itrs+1;
             bool=0;
         end
     end
     if p<eps
         flag(1)=1;
     else
         flag(1)=0;
     end
    resto(2)=norm(Y-ttm(tensor(X),{A,B0,C0}));

    %Gradient descent for B
    gamma=gamma_or;
    itrs=0;
    res=0;
    p=1;
    X_ca=ttm(X,{C0,A},[3,1]);
    X_ca=unfold(tensor(X_ca),2);
    Pinv_ca=pinv(X_ca);
    B=B0;
    Y_term=Y_2*Pinv_ca;
    X_term=X_ca*Pinv_ca;
    bool=1;
     while p>eps && itrs<itrs_max
         B_old=B;
         grad_2=(Y_term-B*X_term);
         B=B+ gamma *grad_2;
         res_old=res;
         res=norm(Y-ttm(X,{A,B,C0}));
         p=res-res_old;
         itrs=itrs+1;
        if (p<eps) || ( itrs>(8/10 *itrs_max)  && (bool==1) )
             gamma=decay*gamma;
             B_old=B;
             grad_2=(Y_term-B*X_term);
             B=B+ gamma *grad_2;
             res_old=res;
             res=norm(Y-ttm(X,{A,B,C0}));
             p=res-res_old;
             itrs=itrs+1;
             bool=0;
         end
     end
      if p<eps
         flag(2)=1;
     else
         flag(2)=0;
      end
    resto(3)=norm(Y-ttm(X,{A,B,C0}));

    %gradient descent for C
    gamma=gamma_or;
    itrs=0;
    res=0;
    p=1;
    X_ba=ttm(X,{B,A},[2,1]);
    X_ba=unfold(tensor(X_ba),3);
    Pinv_ba=pinv(X_ba);
    Y_term=Y_3*Pinv_ba;
    X_term=X_ba*Pinv_ba;
    C=C0;
     while p>eps && itrs<itrs_max
         C_old=C;
         grad_3=(Y_term -C*X_term);
         C=C+ gamma *grad_3;
         res_old=res;
         res=norm(Y-ttm(X,{A,B,C0}));
         p=res-res_old;
         itrs=itrs+1;
         %resto(itrs+itrsA+itrsB)=norm(Y-ttm(tensor(X),{A,B,C0}));
        if (p<eps) || ( itrs>(8/10 *itrs_max)  && (bool==1) )
             gamma=decay*gamma;
             C_old=C;
             grad_3=(Y_term-C*X_term);
             C=C+ gamma *grad_3;
             res_old=res;
             res=norm(Y-ttm(X,{A,B,C0}));
             p=res-res_old;
             itrs=itrs+1;
             bool=0;
         end
      end
      if p<eps
         flag(3)=1;
     else
         flag(3)=0;
     end
     resto(4)=norm(Y-ttm(X,{A,B,C}));
end
