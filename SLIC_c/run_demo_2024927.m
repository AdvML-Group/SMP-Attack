clear; clc;

img_filename = sprintf('datasets/images/%d.png', 657);
%img_filename = sprintf('datasets/%d.png', 644);
img = imread(img_filename);
n=1000;
c=10;
[sp_img, disp_img] = DemoMexSLICSuperpixel (img, n, c);

% 保存为PDF

dispfile = sprintf('disp_%d_%d.png', n, c); % 指定PNG文件名，包含n和c
imwrite(disp_img, dispfile);
spfile, = sprintf('sp_%d_%d.png', n, c); % 指定PNG文件名，包含n和c
imwrite[sp_img, spfile, 