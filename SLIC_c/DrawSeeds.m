function imgSeeds = DrawSeeds(X,Y,img)

X = int64(X);
Y = int64(Y);

[m,n,k] = size(img);

imgSeeds = zeros(m,n);

for i = 1 : length(X)
    imgSeeds(X(i),Y(i))=1;
end
