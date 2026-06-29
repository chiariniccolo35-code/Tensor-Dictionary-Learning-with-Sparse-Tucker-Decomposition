function [U]=overcomplete_dict(U,Mn)
m=size(U,1);
if m<Mn
    j=randi(m,[Mn-m,1]);
    U=[U,U(:,j)];
end
end
