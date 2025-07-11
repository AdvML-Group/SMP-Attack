function [X,Y,L,A,B] = MoveSeeds2LocMinBd (SpBoundary,Lab_img,X,Y,L,A,B,STEP)

[m,n] = size(SpBoundary);

for i=1:length(X)
    Lrow = max([floor(X(i)-STEP*0.25),1]);
    Rrow = min([ceil(X(i)+STEP*0.25),m]);
    Lcol = max([floor(Y(i)-STEP*0.25),1]);
    Rcol = min([ceil(Y(i)+STEP*0.25),n]);
    LocSpBd = SpBoundary(Lrow:Rrow,Lcol:Rcol);
    minMap = imregionalmin(LocSpBd);
    [indX,indY] = meshgrid(Lrow:1:Rrow,Lcol:1:Rcol);
    distMap = sqrt(double(indX-X(i)).^2+double(indY-Y(i)).^2);
    mindistMap = minMap.*distMap;
    mindistMap(mindistMap==0) = 99999.999;
    MinDistValue = min(mindistMap(:));
    [iX,iY] = find(mindistMap == MinDistValue);
    % if multiple mindistvalue are existing, first one is choosed to center
    X(i)= Lrow + iX(1);
    Y(i)= Lcol + iY(1);
    L(i)= Lab_img(X(i),Y(i),1);
    A(i)= Lab_img(X(i),Y(i),2);
    B(i)= Lab_img(X(i),Y(i),3);
end