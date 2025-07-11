function contour = DrawContour3(image,label)
%---------------------------------------------------------------------
% Generate the contour of the label map
%---------------------------------------------------------------------
R=image(:,:,1);
G=image(:,:,2);
B=image(:,:,3);

R(label==1)=255;
G(label==1)=255;
B(label==1)=255;

contour(:,:,1)=R;
contour(:,:,2)=G;
contour(:,:,3)=B;

