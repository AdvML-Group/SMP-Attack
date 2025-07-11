function [X,Y,L,A,B] = PeturbSeeds (EdgeMap,Lab_img,X,Y,L,A,B)
%---------------------------------------------------------------------
% Generate new XYLAB of seeds 
% by moving them to seed locations corresponding to the lowest gradient position (EdgeMap) in a 3x3 neighborhood 
%---------------------------------------------------------------------

CoordinateXY=[-1,-1;-1,0;-1,1;0,-1;0,0;0,1;1,-1;1,0;1,1]; % size:9x2

for i=1:length(X)
    Neighborhood=EdgeMap(X(i)-1:X(i)+1,Y(i)-1:Y(i)+1);
    [value,index]=min(Neighborhood(:));
    X(i)=X(i)+CoordinateXY(index,1);
    Y(i)=Y(i)+CoordinateXY(index,2);
    L(i)=Lab_img(X(i),Y(i),1);
    A(i)=Lab_img(X(i),Y(i),2);
    B(i)=Lab_img(X(i),Y(i),3);
end