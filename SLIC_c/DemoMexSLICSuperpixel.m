function [sp_img,disp_img] = DemoMexSLICSuperpixel (img,K,M)
%---------------------------------------------------------------------------
% Input:
%   img - original rgb image
%   K: number of superpixel
%   M: compactness of superpixel
%   Choice: 1 - constant SLIC ; 2 - Online SLIC ; 3 - maxdistance SLIC
% Output:
%   sp_img - superpixel map
%   disp_img - rgb image with superpixel's boundary
% Function:
%   SLIC Superpixel
%---------------------------------------------------------------------------

%disp('Generating Superpixel ......');

tic;

[CX,CY,CL,CA,CB,Lab_img,STEP] = GetLABXYSeeds(img,K,'NoRandom');
%[CX,CY,CL,CA,CB,Lab_img,STEP] = GetLABXYSeeds(img,K,'Random');

%-----------------------------------------------------------------
% Input variables: <X,Y,L,A,B,STEP,M,K>
L = Lab_img(:,:,1);
A = Lab_img(:,:,2);
B = Lab_img(:,:,3);
STEP=double(STEP);
M=double(M);
K=double(length(CX));  %factual sp number after griding the image

% labels : preliminary superpixel map
% nlabels: final superpixel map in consideration of the connectivity
% contour: the boundaries of the superpixel map

% Tranform Coordinate from Matlab ---> C ; (1,1) ---> (0,0)
CX = CX - 1 ;
CY = CY - 1 ;

[~,enlabels,contour] = SLICSP(CX,CY,CL,CA,CB,L,A,B,STEP,M,K) ;

enlabels = enlabels + 1 ;

Discontour=DrawContour3(img,contour);
sp_img = enlabels;
disp_img = Discontour;  
toc;