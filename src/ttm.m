S = ttm(Y, {F',G',H'} )
%
  %S(M1,M2,M3)=0;
  M1=[];
  M2=[];
  M3=[];
  [M,ii1]=max(abs(double(S)));
  [M,ii2]=max(M);
  [M,ii3]=max(M);
  m3=ii3;
  m2=ii2(m3);
  m1=ii1(m2);
  M1=[M1,m1];
  M2=[M2,m2];
  M3=[M3,m3];
  D1=F(:,M1);
  D2=G(:,M2);
  D3=H(:,M3);