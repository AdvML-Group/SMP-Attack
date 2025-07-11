function [X,Y,L,A,B] = GenRandomInitialCenter (seedx,seedy,STEP,lab_img)
% -----------------------
%  Func:
%    Randomly Generating the initial centers for clustering
% -----------------------
[m,n,k] = size(lab_img);


Scale = 0.4;

% Compute the Range of (X,Y): 90% of STEP
RangeX = [ floor(seedx-STEP*Scale) ; ceil(seedx+STEP*Scale) ];
RangeY = [ floor(seedy-STEP*Scale) ; ceil(seedy+STEP*Scale) ];

% Compute the randomized center
RandSeeds = rand(1);
X = RangeX(1)+RandSeeds*(RangeX(2)-RangeX(1));
RandSeeds = rand(1);
Y = RangeY(1)+RandSeeds*(RangeY(2)-RangeY(1));
% set limitation: [2,2]~[m-1,n-1]
X = min([max([X,2]),m-1]);
Y = min([max([Y,2]),n-1]);

% Compute the vector [l,a,b] of randomized center (X,Y), 
% with width of neighbourhood [7,7]

% set limitation: [1,1]~[m,n]
X_left = max([X-3,1]);
Y_left = max([Y-3,1]);
X_right = min([X+3,m]);
Y_right = min([Y+3,n]);

imgNB_L = lab_img(X_left:X_right,Y_left:Y_right,1);
imgNB_A = lab_img(X_left:X_right,Y_left:Y_right,2);
imgNB_B = lab_img(X_left:X_right,Y_left:Y_right,3);

L=mean(imgNB_L(:));
A=mean(imgNB_A(:));
B=mean(imgNB_B(:));


