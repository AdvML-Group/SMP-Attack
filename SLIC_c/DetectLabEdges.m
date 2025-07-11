function EdgeMap =  DetectLabEdges(Lab_img)
%---------------------------------------------------------------------
% Generate the edge map of Lab_img
%---------------------------------------------------------------------
[m,n,k]=size(Lab_img);
EdgeMap=zeros(m,n);

for i=2:m-1
    for j=2:n-1
        dx=sum((Lab_img(i,j-1)-Lab_img(i,j+1)).^2);
        dy=sum((Lab_img(i-1,j)-Lab_img(i+1,j)).^2);
        EdgeMap(i,j)=dx^2+dy^2;
    end
end