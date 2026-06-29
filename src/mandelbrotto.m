clear
% Parameters for the tensor
rows = 100;     % Number of rows
cols = 100;     % Number of columns
num_slices = 10; % Number of zoom levels (depth of tensor)

% Initialize the 3D tensor
mandelbrot_tensor = zeros(rows, cols, num_slices);

% Generate the Mandelbrot set for each zoom level
for slice = 1:num_slices
    % Define zoom level
    zoom_factor = 1.15^(slice - 1); % Exponential zoom factor
    x = linspace(-2/zoom_factor, 1/zoom_factor, cols);
    y = linspace(-1.5/zoom_factor, 1.5/zoom_factor, rows);
    [X, Y] = meshgrid(x, y);
    Z = X + 1i * Y;
    C = Z;
    
    % Initialize Mandelbrot calculation
    mandelbrot = zeros(rows, cols);
    max_iter = 30; % Maximum number of iterations
    
    for k = 1:max_iter
        Z = Z.^2 + C;
        mandelbrot(abs(Z) < 2) = k;
    end
    
    % Store the Mandelbrot set in the tensor slice
    mandelbrot_tensor(:, :, slice) = mandelbrot;
end

I1=100;
I2=100;
I3=10;
M5=floor(I1*1.5);
M6=floor(I2*1.5);
M7=floor(I3*1.5);
% [F,S,V]=svds(Y_1,I1);
% %F=overcomplete_dict(F,M5);
% [G,S,V]=svds(Y_2,I2);
% %G=overcomplete_dict(G,M6);
% [H,S,V]=svds(Y_3,I3);
 % A_0 = nvecs(tensor(mandelbrot_tensor),1,I1); % M_{1} left singular vectors of Y_{1} % In this way should be right
 % A_0=overcomplete_dict(A_0,M5);
 % B_0 = nvecs(tensor(mandelbrot_tensor),2,I2); % M_{2} left singular vectors of Y_{2}
 % B_0=overcomplete_dict(B_0,M6);
 % C_0 = nvecs(tensor(mandelbrot_tensor),3,I3); 
 % C_0=overcomplete_dict(C_0,M7);
%H=overcomplete_dict(H,M7);
A_0=randn(I1,M5);
B_0=randn(I2,M6);
C_0=randn(I3,M7);
%% GRADTENSOR ITERATIONS
% 
%  [X_1, A, B, C,residual] = GRAD_TENSOR(mandelbrot_tensor,1100,0.1,1e-6,1e-5,A_0,B_0,C_0,10,1000,0.1);
%  mandel_est=ttm(X_1,{A,B,C});
% for slice = 1:num_slices
%     figure;
%     title(['Mandelbrot Set - Slice ' num2str(slice) ' (Zoom Factor: ' num2str(1.15^(slice-1)) ')']);
%     subplot(1,2,1)
%     imagesc(mandelbrot_tensor(:, :, slice));
%     colormap parula;
%     axis image;
%     colorbar;
%     subplot(1,2,2)
%     imagesc(double(mandel_est(:, :, slice)));
%     colormap parula;
%     axis image;
%     colorbar;
% end
% Data=[];
% 
% for tmax=10:500:10000
%     [X_1, A, B, C,residual] = GRAD_TENSOR(mandelbrot_tensor,tmax,0.1,1e-6,1e-5,A_0,B_0,C_0,10,1000,0.1);
%     Data=[Data;residual(end)/norm(tensor(mandelbrot_tensor))];
% end
% figure(1)
% plot(Data)
%%
[X_1, A, B, C,residual] = GRAD_TENSOR(mandelbrot_tensor,7000,0.3,1e-6,1e-5,A_0,B_0,C_0,10,1000,0.1);
mandel_est=ttm(X_1,{A,B,C});
for slice = 1:num_slices
    figure;
    title(['Mandelbrot Set - Slice ' num2str(slice) ' (Zoom Factor: ' num2str(1.15^(slice-1)) ')']);
    subplot(1,2,1)
    imagesc(mandelbrot_tensor(:, :, slice));
    colormap parula;
    axis image;
    colorbar;
    subplot(1,2,2)
    imagesc(double(mandel_est(:, :, slice)));
    colormap parula;
    axis image;
    colorbar;
end
%%
figure(2)
Data=[];
[X_1, A, B, C,residual] = GRAD_TENSOR(mandelbrot_tensor,9000,0.35,1e-6,1e-5,A_0,B_0,C_0,10,1000,0.1);
plot(residual/norm(tensor(mandelbrot_tensor)))