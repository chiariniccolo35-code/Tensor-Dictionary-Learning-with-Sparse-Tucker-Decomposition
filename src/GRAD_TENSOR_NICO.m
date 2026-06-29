
function [A, B, C,residual] = GRAD_TENSOR_NICO(Y,X,gamma,eps_1,eps_2,itrs_max)

% INPUT
  % Y is the tensor given in input;
  % X is the sparse core tensor (obtained via tensor toolbox);
  % s is the maximum sparsity value (total number of non-zeros);
  % gamma is the step size;
  % eps_1,eps_2 are tolerance parameters;

% OUTPUT
  % A, B, C are the dictionary updated;

  X = tensor(X);

  % Size of the tensor Y and the core tensor X
    I = size(Y); M = size(X);
    I1 = I(1); I2 = I(2); I3 = I(3);
    M1 = M(1); M2 = M(2); M3 = M(3);
     
  % Inizialization of the disctionaries ( double type)
    A_0 = nvecs(tensor(Y),1,I1); % M_{1} left singular vectors of Y_{1} % In this way should be right
    B_0 = nvecs(tensor(Y),2,I2); % M_{2} left singular vectors of Y_{2}
    C_0 = nvecs(tensor(Y),3,I3); % M_{3} left singular vectors of Y_{3}

  % We define the mode-1,2,3 for the tensor Y and for the core tensor (
  % double type)
    Y_1 = unfold(tensor(Y),1); X_1 = unfold(X,1);
    Y_2 = unfold(tensor(Y),2); X_2 = unfold(X,2);
    Y_3 = unfold(tensor(Y),3); X_3 = unfold(X,3);

  
  % We compute the gradient for A
    z_A  = ttm(permute(X,[2 3 1]),{B_0,C_0},[1,2]); zr_A = reshape(z_A,[M1,I2*I3]);
    %z_A  = ttm(X,{B_0,C_0},[1,2]); zr_A = reshape(z_A,[M1,I2*I3]);
    Grad_A = ( Y_1 - A_0 * double(zr_A) ) * pinv( double(zr_A) );
    size(Grad_A)

  % We compute the gradient for B
    z_B  = ttm(permute(X,[1,3,2]),{A_0,C_0},[1,2]);   zr_B = reshape(z_B,[M2,I3*I1]);
    
    Grad_B = ( Y_2 - B_0 * double(zr_B) ) * pinv( double(zr_B) );

  % We compute the gradient for C
    z_C = ttm(permute(X,[1,2,3]),{A_0,B_0},[1,2]); zr_C = reshape(z_C,[M3,I1*I2]);

    Grad_C = ( Y_3 - C_0 * double(zr_C) ) * pinv( double(zr_C) );
     
  % Cycle that stops when quantity is less than eps_2
    residual = norm( tensor(Y) - ttm(X,{A_0,B_0,C_0}) )^2;

    %max_iter = 100;
    %iter = 1;

    %while( (residual > eps_2) ) %&& (iter< max_iter))
        %

        %iter = iter + 1;

        % We compute A until the residual is less than eps_1
        A  = A_0 + gamma * Grad_A;
        residual_A = norm(A-A_0,'fro')^2;
        itrs=0;
        while(residual_A > eps_1) && (itrs<itrs_max)
            %
            A_0 = A;
            Grad_A = ( Y_1-A_0 * double(zr_A) ) * pinv( double(zr_A) );
            A = A_0 + gamma * Grad_A;
            residual_A = norm(A-A_0,'fro')^2;
            itrs=itrs+1;
        end

        % We compute B until the residual is less than eps 1
        B  = B_0 + gamma * Grad_B;
        residual_B = norm(B-B_0,'fro')^2;
        itrs=0;
        while(residual_B > eps_1) && (itrs<itrs_max)
            %
            B_0 = B;
            Grad_B = ( Y_2 - B_0 * double(zr_B) ) * pinv( double(zr_B) );
            B  = B_0 + gamma * Grad_B;
            residual_B = norm(B-B_0,'fro')^2;
            itrs=itrs+1;
        end

        % We compute C until the residual is less than eps_2
        C  = C_0 + gamma * Grad_C;
        residual_C = norm(C-C_0,'fro')^2;
        itrs=0;
        while(residual_C > eps_1) && (itrs<itrs_max)
            %
            C_0 = C;
            Grad_C = ( Y_3 - C_0 * double(zr_C) ) * pinv( double(zr_C) );
            C  = C_0 + gamma * Grad_C;
            residual_C = norm(C-C_0,'fro')^2;    
            itrs=itrs+1;
        end

        A_0 = A;
        B_0 = B;
        C_0 = C;
        residual = norm( tensor(Y) - ttm(X,{A_0,B_0,C_0}) )^2;
    end

