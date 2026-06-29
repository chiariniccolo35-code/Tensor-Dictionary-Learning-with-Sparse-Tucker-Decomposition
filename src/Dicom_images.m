clear all; clc; close all;

% Il set di immagine mediche formato DICOM è costituito da 10 cartelle
% ciascuna espressa attraverso un numero , la prima 00 e l'ultima 90.

  a = input('Insert the index of the medical image that we want to analyze the index have to be between 0 and 9: ');
  while( (a <=-1) || (a > 9))
      fprintf('\n Index not acceptable,');
      a = input(' reinsert: ');
  end

% Number converted into a string
  a = num2str(a);

% Construction of the path
  filepattern = sprintf('4DCT-Dicom/%s0',a);

% Extraction of the volume image selected
  V = dicomreadVolume('90'); V = squeeze(V);
  V = imresize3(V,[151, 125, 141]);
  V = rescale(V, 0, 1);

  I = size(V);
  I1 = I(1); I2 = I(2); I3 = I(3);

% We apply hosvd to our tensor to obatin the three disctionaries;
  T = hosvd(tensor(double(V)),5e-10);

  % Dictionaries
    A = T.U{1}; B = T.U{2}; % Dimension have to be imolemented??
    C = T.U{3};

% Setting of the parameters
  eps = 10^-2; itermax = 500; tmax = 10^4;

% TOMP applied to our tensor given
  [E,m1,m2,m3,r] = efficientTOMP2(A,B,C,V,tmax,eps,itermax);
  % New core tensor is
    %X_1 = E;
    X = sptensor([],[],[size(A,2), size(B,2), size(C,2)]); 
    E = sptensor(E);
    X(m1,m2,m3) = E;

 % GRADTENSOR

   % Setting of the oarameters
    eps_1 = 10^-6; eps_2 = 10^-4; gamma = 0.3;
    [A1,B1,C1,residual] = GRAD_TENSOR_NICO(V,X,gamma,eps_1,eps_2,500)
   
  





