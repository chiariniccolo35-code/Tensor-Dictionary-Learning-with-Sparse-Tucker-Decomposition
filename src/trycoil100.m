clear
n1=128;n2=128;ne=71;np=60; %approx np=60
adr='/home/alberto/matlabScripts/matrix/project/4DCT-Dicom/';   % To be changed as needed
 %ob=51;
% ob=27;
 %ob=4;
% ob=14;
ob=54;
%ob=83 ;  %pera
%ob=23;   %macchinina
k1=0;
for k=5:5:300
   k1=k1+1;
   A=imread([adr,'coil-100/obj',int2str(ob),'__',num2str(k),'.png']); 

   Y(:,:,:,k1)=A;
end
%provare a normalizzare
%%
I=size(Y);
M=1.4*I;
Yd=double(Y)/253;
Yt=tensor(Yd);
A_0=nvecs(Yt,1,I(1));
A_0=overcomplete_dict(A_0,round(M(1)));
B_0=nvecs(Yt,2,I(2));
B_0=overcomplete_dict(B_0,round(M(2)));
C_0=nvecs(Yt,3,I(3));
C_0=overcomplete_dict(C_0,round(M(3)));
D_0=nvecs(Yt,4,I(4));
D_0=overcomplete_dict(D_0,round(M(4)));

max_iter=10;
itrs2=1000;
%%
[X, A, B, C, D, residual] = GRAD_TENSOR4D(Yd,1e6,0.2,4e-6,3e-6,A_0,B_0,C_0,D_0,10,itrs2,0.9);
%%
Y_est=253*ttm(X,{A,B,C,D});
Y_est=cast(double(Y_est),"uint8");
%%
subplot(1,2,1)
imshow((Y(:,:,:,3)))
subplot(1,2,2)
imshow(Y_est(:,:,:,3))

%casetta res 0.12
%pera 0.08