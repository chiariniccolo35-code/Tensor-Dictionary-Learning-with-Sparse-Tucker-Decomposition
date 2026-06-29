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

% [E,M1,M2,M3,r1]=efficientTOMP(F,G,H,P,30000,1e-5,500);
% X=sptensor([],[],[M5,M6,M7]);
% E3=sptensor(E);
% X(M1,M2,M3)=E3;
[E2,M12,M22,M32,r2]=efficientTOMP2(F,G,H,P,30000,1e-5,500);
X2=sptensor([],[],[M5,M6,M7]);
E4=sptensor(E2);
X2(M12,M22,M32)=E4;

%X=tensor(X);
fprintf('errore dopo primo TOMP: %f \n',norm(Y-ttm(X2,{F,G,H})));
[A,B,C,X_cb,A0,B0,C0,resto1]=GradTensor(Y,tensor(X2),0.3,1e-6,F,G,H,500);
fprintf('errore dopo primo GRADTENSOR: %f \n',norm(Y-ttm(X2,{A,B,C})));
fprintf('\n');
for i=1:5
    % [E,M1,M2,M3,r]=efficientTOMP(A,B,C,P,30000,1e-5,1000);
    % X=sptensor([],[],[M5,M6,M7]);
    % E=sptensor(E);
    % X(M1,M2,M3)=E;
    [E2,M1,M2,M3,r2]=efficientTOMP2(A,B,C,P,30000,1e-5,500);
    X2=sptensor([],[],[M5,M6,M7]);
    E2=sptensor(E2);
    X2(M1,M2,M3)=E2;
   % X=tensor(X);
    [A,B,C,X_cb,A0,B0,C0,resto3]=GradTensor(Y,tensor(X2),0.1,0.0001,A,B,C,500);
    l(i)=norm(Y-ttm(X2,{A,B,C}));
end