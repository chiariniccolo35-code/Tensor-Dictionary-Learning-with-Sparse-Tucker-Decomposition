function [X, A, B, C,D,residual] = GRAD_TENSOR4D(Y,tmax,gamma,eps_1,eps_2,A_0,B_0,C_0,D_0,max_iter,itrs2,decay)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUT
%%%% Y: input tensor;
%%%% s: maximum sparsity value (total number of non-zeros);
%%%% gamma: step size for Gradient Descent;
%%%% eps_1,eps_2: tolerance parameters;
%%%% A_0, B_0, C_0: input dictionaries
%%%% max_iter: maximum number of iterations for GD
%%%% itrs2: maximum number of iterations for TOMP
% OUTPUT
%%%% A, B, C: the updated dictionary;
%%%% X: sparse core tensor;
%%%% residual: vector of residuals residual(i) residual after ith iter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


  % Setting the parameters

  % Size of the tensor Y and the core tensor X
    M1 = size(A_0,2); M2 = size(B_0,2); M3 = size(C_0,2); M4=size(D_0,2);
     

  % We define the mode-1,2,3 for the tensor Y and for the core tensor (
  % double type)
    Y_1 = unfold(tensor(Y),1);  %X_1 = unfold(X,1);
    Y_2 = unfold(tensor(Y),2);  %X_2 = unfold(X,2);
    Y_3 = unfold(tensor(Y),3);  %X_3 = unfold(X,3);
    Y_4=  unfold(tensor(Y),4);
  
  % We compute the gradient for A
  
    
     
  % Cycle that stops when quantity is less than eps_2
    residual(1) = 1;

    iter = 0;

    while( (residual(iter+1) > eps_2)  && (iter< max_iter))
        %TOMP iteration
          [E,m1,m2,m3,m4,r] = efficientTOMP4d(A_0,B_0,C_0,D_0,Y,tmax,eps_2,itrs2);
          X = sptensor([],[],[M1,M2,M3,M4]);
          E = sptensor(E);
          X(m1,m2,m3,m4) = E;
          if iter+1==1
              residual(1)=norm(Y-ttm(X,{A_0,B_0,C_0,D_0}));
          end
  % We compute the gradient for A

          z_A  = ttm(X,{B_0,C_0,D_0},[2,3,4]); zr_A = unfold(tensor(z_A),1);
          Grad_A = ( Y_1 - A_0 * double(zr_A) ) * pinv( double(zr_A) );
   

  % We compute the gradient for B

          z_B  = ttm(X,{A_0,C_0,D_0},[1,3,4]); zr_B = unfold(tensor(z_B),2);
    
          Grad_B = ( Y_2 - B_0 * double(zr_B) ) * pinv( double(zr_B) );

  % We compute the gradient for C

          z_C = ttm(X,{A_0,B_0,D_0},[1,2,4]); zr_C = unfold(tensor(z_C),3);

          Grad_C = ( Y_3 - C_0 * double(zr_C) ) * pinv( double(zr_C) );

          z_D= ttm(X,{A_0,B_0,C_0},[1,2,3]); zr_D= unfold(tensor(z_D),4);

          Grad_D = ( Y_4 - D_0 * double(zr_D )) * pinv( double(zr_D) );


%Gradient descent iterations for A

          A  = A_0 + gamma * Grad_A;
          residual_A = norm(A-A_0,'fro')^2;
          prod_A_2 = double(zr_A) * pinv( double(zr_A) );
          prod_A_1 = Y_1 * pinv( double(zr_A) );
          gamma_A = gamma;
          gamma_B = gamma;
          gamma_C = gamma;
          gamma_D = gamma;

          while(residual_A > eps_1)
              
              A_0 = A;
              Grad_A = prod_A_1 - A_0 * prod_A_2;
              A = A_0 + gamma_A * Grad_A;
              residual_A = norm(A-A_0,'fro')^2;
              if residual_A<eps_1
                  gamma_A=decay*gamma_A;
                  A_0 = A;
                  Grad_A = prod_A_1 - A_0 * prod_A_2;
                  A = A_0 + gamma_A * Grad_A;
                  residual_A = norm(A-A_0,'fro')^2;

              end
          end

%Gradient descent iterations for B

          B  = B_0 + gamma_B * Grad_B;
          residual_B = norm(B-B_0,'fro')^2;
          prod_B_1 = Y_2 * pinv( double(zr_B) );
          prod_B_2 = double(zr_B) * pinv( double(zr_B) );
          while(residual_B > eps_1)
              
              B_0 = B;
              Grad_B = ( prod_B_1 - B_0 * prod_B_2);
              B  = B_0 + gamma_B * Grad_B;
              residual_B = norm(B-B_0,'fro')^2;
              if residual_B <eps_1
                  gamma_B=decay*gamma_B;
                   B_0 = B;
                  Grad_B = ( prod_B_1 - B_0 * prod_B_2);
                  B  = B_0 + gamma_B * Grad_B;
                  residual_B = norm(B-B_0,'fro')^2;
              end
          end
          
 %Gradient descentiterations for C
          
          C  = C_0 + gamma_C * Grad_C;
          residual_C = norm(C-C_0,'fro')^2;
          prod_C_1 = Y_3 * pinv( double(zr_C) );
          prod_C_2 = double(zr_C) * pinv( double(zr_C) );
          while(residual_C > eps_1)
              C_0 = C;
              Grad_C = ( prod_C_1 - C_0 * prod_C_2);
              C  = C_0 + gamma_C * Grad_C;
              residual_C = norm(C-C_0,'fro')^2;   
              if residual_C<eps_1
                  gamma_C=decay*gamma_C;
                   C_0 = C;
                   Grad_C = ( prod_C_1 - C_0 * prod_C_2);
                   C  = C_0 + gamma_C * Grad_C;
                   residual_C = norm(C-C_0,'fro')^2;  
              end
          end

           %Gradient descent iterations for A

          D  = D_0 + gamma * Grad_D;
          residual_D = norm(D-D_0,'fro')^2;
          prod_D_2 = double(zr_D) * pinv( double(zr_D) );
          prod_D_1 = Y_4 * pinv( double(zr_D) );
          while(residual_D > eps_1)
              
              D_0 = D;
              Grad_D = prod_D_1 - D_0 * prod_D_2;
              D = D_0 + gamma_D * Grad_D;
              residual_D = norm(D-D_0,'fro')^2;
              if residual_D<eps_1
                  gamma_D=decay*gamma_D;
                  D_0 = D;
                  Grad_D = prod_D_1 - D_0 * prod_D_2;
                  D = D_0 + gamma_D * Grad_D;
                  residual_D = norm(D-D_0,'fro')^2;

              end
          end 


          A_0 = A;
          B_0 = B;
          C_0 = C;
          D_0 = D;
          iter = iter + 1;
          if iter>7/10* max_iter
              gamma=gamma*decay;
          end
          residual(iter+1) = norm( Y - ttm(X,{A_0,B_0,C_0,D_0}) );
         
    end



    















     
  

