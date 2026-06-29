clear
% Parameters
I1 = 100; I2 = 100; I3 = 100; % Tensor dimensions
M1 = round(1.5 * I1); % Core tensor dimensions
M2 = round(1.5 * I2);
M3 = round(1.5 * I3);
sparsity_ratio = 1/6; % Sparsity level

%% Step 1: Generate random dictionaries (A, B, C) - leading singular vectors
%Y = sptenrand([I1, I2, I3],0.8*prod([I1,I2,I3])); % Random input tensor

%% GRADTENSOR ITERATIONS
% for i=1:10
%
    % [E,m1,m2,m3,r]=efficientTOMP2(A,B,C,Y,156250,1e-5,1000);
    %  X=sptensor([],[],[M1,M2,M3]);
    %  E=sptensor(E);
    %  X(m1,m2,m3)=E;
    %  norm(Y-ttm(X,{A,B,C}))

     %%
%     [A,B,C,flag,resto]=GradTensorEff(tensor(Y),X,0.3,1e-7,A,B,C,1000,0.01);
%     res(i)=norm(Y-ttm(X,{A,B,C}))/norm(Y);
%     Data=[Data;r,resto];
% end
% 
% [E,M1,M2,M3,r]=efficientTOMP2(A_0,B_0,C_0,Y,7300,1e-5,1000);
%      X=sptensor([],[],[M5,M6,M7]);
%      E=sptensor(E);
%      X(M1,M2,M3)=E;
 %norm(Y-ttm(X,{A,B,C}))
 %%
Data=[];
for i=1:105
    A_0=randn(I1,M1);
    B_0=randn(I2,M2);
    C_0=randn(I3,M3);
    X=sptenrand([M1,M2,M3],1/3^3 );
    Y=ttm(X,{A_0,B_0,C_0});
    [X_1, A, B, C,residual] = GRAD_TENSOR(Y,450000,0.3,3e-6,1e-5,A_0,B_0,C_0,30,1000,0.5);
    Data=[Data;residual];
end
%% plot residue
figure(1)
plot(mean(Data)/norm(Y))
xlabel('number of iterations')
ylabel('relative residual')
%%
    A_0=randn(I1,M1);
    B_0=randn(I2,M2);
    C_0=randn(I3,M3);
    X=sptenrand([M1,M2,M3],1/3^3 );
    Y=ttm(X,{A_0,B_0,C_0});
    Data2=[];
    tmax=linspace(1e4,1e6,10);
    tmax=[1000,tmax]
    %%
for i=1:length(tmax)
    [X_1, A, B, C,residual] = GRAD_TENSOR(Y,tmax(i),0.3,3e-6,1e-5,A_0,B_0,C_0,30,1000,0.5);
    Data2=[Data2;residual(end)];
end
%%
[X_1, A, B, C,residual] = GRAD_TENSOR(Y,2e6,0.3,3e-6,1e-5,A_0,B_0,C_0,30,1000,0.5);
Data2=[Data2;residual(end)];
%%
figure(2)
axis tight
plot([tmax,1.5e6,2e6],Data2/norm(Y))
xlabel('tmax')
ylabel('residual relative error')


