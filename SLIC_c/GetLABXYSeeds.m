function [X,Y,L,A,B,Lab_img,STEP] = GetLABXYSeeds(img,K,choice)
%---------------------------------------------------------------------
% Generate the seeds of superpixel and Extract the XYLab information of seeds
%
% Choice: 1 - centroid of square ; 2 - randomized centers ;
%---------------------------------------------------------------------

% img is the original image
% K is the number of superpixel
Lab_img=colorspace('RGB->Lab',img);
% Lab_img is the color image in CIE L*a*b space

[m,n,k]=size(Lab_img);


% superpixelSize=int64(0.5+double(m*n)/double(K));
superpixelSize=int64(double(m*n)/double(K));
% STEP=int64(sqrt(double(superpixelSize))+0.5);
STEP=int64(sqrt(double(superpixelSize)));
% xstrips=int64(0.5+double(m)/double(STEP));
xstrips=int64(double(m)/double(STEP));
% ystrips=int64(0.5+double(n)/double(STEP));
ystrips=int64(double(n)/double(STEP));
xerr=m-STEP*xstrips;
if xerr < 0
    xstrips=xstrips-1;
    xerr = m-STEP*xstrips;
end
yerr=n-STEP*ystrips;
if yerr < 0
    ystrips=ystrips-1;
    yerr=n-STEP*ystrips;
end
xerrperstrip = double(xerr)/double(xstrips);
yerrperstrip = double(yerr)/double(ystrips);
xoff=int64(STEP/2);
yoff=int64(STEP/2);

numseeds = xstrips*ystrips;
X=zeros(numseeds,1);
Y=zeros(numseeds,1);
L=zeros(numseeds,1);
A=zeros(numseeds,1);
B=zeros(numseeds,1);

n=1;
for y=0:ystrips-1
    ye=int64(y*yerrperstrip);
    for x=0:xstrips-1
        xe=int64(x*xerrperstrip);
        seedx=int64(x*STEP+xoff+xe);
        seedy=int64(y*STEP+yoff+ye);
        if strcmp(choice,'Random') % Insert Randomized Centers
            [X(n),Y(n),L(n),A(n),B(n)] = GenRandomInitialCenter (seedx,seedy,STEP,Lab_img);
        else % centroid of square 
            X(n)=seedx;
            Y(n)=seedy;
            L(n)=Lab_img(seedx,seedy,1);
            A(n)=Lab_img(seedx,seedy,2);
            B(n)=Lab_img(seedx,seedy,3);
        end
        n=n+1;
    end
end

    
