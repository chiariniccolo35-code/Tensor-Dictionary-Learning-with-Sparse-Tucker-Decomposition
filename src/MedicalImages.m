%% Loading Data and initialize dictionaries
clear
A=dicomreadVolume('40');
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
% [F,S,V]=svds(Y_1,I1);
% %F=overcomplete_dict(F,M5);
% [G,S,V]=svds(Y_2,I2);
% %G=overcomplete_dict(G,M6);
% [H,S,V]=svds(Y_3,I3);
 A_0 = nvecs(tensor(Y),1,I1); % M_{1} left singular vectors of Y_{1} % In this way should be right
 A_0=overcomplete_dict(A_0,M5);
 B_0 = nvecs(tensor(Y),2,I2); % M_{2} left singular vectors of Y_{2}
 B_0=overcomplete_dict(B_0,M6);
 C_0 = nvecs(tensor(Y),3,I3); 
 C_0=overcomplete_dict(C_0,M7);
%H=overcomplete_dict(H,M7);
%% GRADTENSOR ITERATIONS

[X_1, A, B, C,residual] = GRAD_TENSOR(P,837000,0.35,1e-6,1e-5,A_0,B_0,C_0,10,100,0.01);
%837000
%% Display obtained results
y_est=ttm(X_1,{A,B,C});
figure(1)
subplot(1,2,1)
imshow(double(Y(:,:,50)))
title('original 50th slice')
subplot(1,2,2)
imshow(double(y_est(:,:,50)));
title('reconstructed 50th slice')
%% Residual
residual
%% Different sparsity level
[X_2, A, B, C,residual] = GRAD_TENSOR(P,330000,0.35,1e-6,1e-5,A_0,B_0,C_0,10,100,0.01);
y_est2=ttm(X_2,{A,B,C});
%%
figure(1)
subplot(1,3,1)
imshow(double(Y(:,:,50)))
title('original 50th slice')
subplot(1,3,2)
imshow(double(y_est(:,:,50)));
title('reconstructed with 31% nnz')
subplot(1,3,3)
imshow(double(y_est2(:,:,50)));
title('reconstructed with 12% nnz')

