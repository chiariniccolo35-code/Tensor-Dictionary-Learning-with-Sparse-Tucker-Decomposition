clear
A=dicomreadVolume('90');
A=squeeze(A);
A=imresize3(A,[151,125,141]);
A=rescale(A,0,1);
P=A;
Y=tensor(A);
Y_1=unfold(Y,1);
Y_2=unfold(Y,2);
Y_3=unfold(Y,3);
I1=151;
I2=125;
I3=141;
M5=floor(151*1.5);
M6=floor(I2*1.5);
M7=floor(I3*1.5);
[F,S,V]=svds(Y_1,I1);
F=overcomplete_dict(F,M5);
[G,S,V]=svds(Y_2,I2);
G=overcomplete_dict(G,M6);
[H,S,V]=svds(Y_3,I3);
H=overcomplete_dict(H,M7);

[E,M1,M2,M3,r1]=efficientTOMP(F,G,H,P,30000,1e-5,500);
X=sptensor([],[],[M5,M6,M7]);
E3=sptensor(E);
X(M1,M2,M3)=E3;
[E2,M12,M22,M32,r2]=efficientTOMP2(F,G,H,P,30000,1e-5,500);
X2=sptensor([],[],[M5,M6,M7]);
E4=sptensor(E2);
X2(M12,M22,M32)=E4;

%X=tensor(X);
fprintf('errore dopo primo TOMP: %f \n',norm(Y-ttm(X2,{F,G,H})));
[Ak,Bk,Ck,X_cb,A0,B0,C0,resto1]=GradTensor(Y,tensor(X),0.3,1e-6,F,G,H,500);
[A,B,C,X_cb,A0,B0,C0,resto1]=GradTensor(Y,X2,0.3,1e-6,F,G,H,500);
fprintf('errore dopo primo GRADTENSOR: %f \n',norm(Y-ttm(X2,{A,B,C})));
fprintf('\n');
fprintf('errore tra tensors dopo primo GRADTENSOR: %f   %f \n',norm(X-X2), norm(X-tensor(X2)));
%%
%check_tensor_operations(Y, tensor(X), F, G, H,100)

    % X_cb2=ttm(X,{H,G},[3,2]);
    % X_cb2=unfold(tensor(X_cb2),1);
    % S=ttm(X,{A0,B0,C0});
    % norm(unfold(S,1)-A0*X_cb)
    % norm(X_cb-X_cb2)
[E,M1,M2,M3,r2]=efficientTOMP(Ak,Bk,Ck,P,30000,1e-5,1000);
X=sptensor([],[],[M5,M6,M7]);
E=sptensor(E);
X(M1,M2,M3)=E;
[E2,M1,M2,M3,r4]=efficientTOMP2(A,B,C,P,300000,1e-5,500);
X2=sptensor([],[],[M5,M6,M7]);
E2=sptensor(E2);
X2(M1,M2,M3)=E2;
fprintf('errore dopo secondo TOMP: %f \n',norm(Y-ttm(X2,{A,B,C})));

[A2k,B2k,C2k,X_cb,A0,B0,C0,resto2]=GradTensor(Y,tensor(X),0.3,1e-6,Ak,Bk,Ck,500);
[A2,B2,C2,X_cb,A0,B0,C0,resto1]=GradTensor(Y,X2,0.3,1e-6,Ak,Bk,Ck,500);

fprintf('errore dopo secondo GRADTENSOR: %f \n',norm(Y-ttm(X2,{A2,B2,C2})));
fprintf('\n');

fprintf('errore dopo secondo GRADTENSOR tra tensors: %f %f\n',norm(X-X2), norm(X-tensor(X2)));
fprintf('\n');
%%

[E,M1,M2,M3,r]=efficientTOMP(A2k,B2k,C2k,P,30000,1e-5,1000);
X=sptensor([],[],[M5,M6,M7]);
E=sptensor(E);
X(M1,M2,M3)=E;
[E2,M1,M2,M3,r2]=efficientTOMP2(A2,B2,C2,P,30000,1e-5,500);
X2=sptensor([],[],[M5,M6,M7]);
E2=sptensor(E2);
X2(M1,M2,M3)=E2;
fprintf('errore dopo terzo TOMP: %f \n',norm(Y-ttm(X,{A2k,B2k,C2k})));
[A3k,B3k,C3k,X_cb,A0,B0,C0,resto3]=GradTensor(Y,tensor(X),0.3,1e-6,A2k,B2k,C2k,500);
[A3,B3,C3,X_cb,A0,B0,C0,resto1]=GradTensor(Y,X2,0.3,1e-6,A2,B2,C2,500);
fprintf('errore dopo terzo GRADTENSOR: %f \n',norm(Y-ttm(X2,{A3,B3,C3})));
fprintf('\n');
fprintf('errore dopo secondo GRADTENSOR tra tensors: %f %f\n',norm(X-X2), norm(X-tensor(X2)));
fprintf('\n');
%%

[E,M1,M2,M3,r]=efficientTOMP(A3k,B3k,C3k,P,30000,1e-5,1000);
X=sptensor([],[],[M5,M6,M7]);
E=sptensor(E);
X(M1,M2,M3)=E;
[E2,M1,M2,M3,r2]=efficientTOMP2(A3,B3,C3,P,30000,1e-5,500);
X2=sptensor([],[],[M5,M6,M7]);
E2=sptensor(E2);
X2(M1,M2,M3)=E2;
%X=tensor(X);
fprintf('errore dopo quarto TOMP: %f \n',norm(Y-ttm(tensor(X),{A3k,B3k,C3k})));
[A4k,B4k,C4k,X_cb,A0,B0,C0,resto3]=GradTensor(Y,tensor(X),0.3,1e-6,A3k,B3k,C3k,500);
[A4,B4,C4,X_cb,A0,B0,C0,resto1]=GradTensor(Y,X2,0.3,1e-6,A3,B3,C3,500);
fprintf('errore dopo quarto GRADTENSOR: %f \n',norm(Y-ttm(X2,{A4,B4,C4})));
fprintf('\n');
fprintf('errore dopo secondo GRADTENSOR tra tensors: %f %f\n',norm(X-X2), norm(X-tensor(X2)));
fprintf('\n');
%%
for i=1:5
    % [E,M1,M2,M3,r]=efficientTOMP(A4k,B4k,C4k,P,30000,1e-5,1000);
    % X=sptensor([],[],[M5,M6,M7]);
    % E=sptensor(E);
    % X(M1,M2,M3)=E;
    [E2,M1,M2,M3,r2]=efficientTOMP2(A4,B4,C4,P,30000,1e-5,500);
    X2=sptensor([],[],[M5,M6,M7]);
    E2=sptensor(E2);
    X2(M1,M2,M3)=E2;
    [A4,B4,C4,X_cb,A0,B0,C0,resto3]=GradTensor(Y,X2,0.1,0.0001,A4,B4,C4,500);
    l(i)=norm(Y-ttm(X2,{A4,B4,C4}));
end