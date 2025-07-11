#[SLIC superpixels paper](http://www.kevsmith.com/papers/SLIC_Superpixels.pdf）

import math
from skimage import io, color, transform
import numpy as np
from tqdm import trange
import argparse


class Cluster(object):
    cluster_index = 1

    def __init__(self, h, w, l=0, a=0, b=0):
        # 初始化聚类簇的像素点和编号
        self.update(h, w, l, a, b)
        self.pixels = []
        self.no = self.cluster_index
        Cluster.cluster_index += 1

    def update(self, h, w, l, a, b):
        # 更新聚类簇的信息
        self.h = h
        self.w = w
        self.l = l
        self.a = a
        self.b = b

    def __str__(self):
        return "{},{}:{} {} {} ".format(self.h, self.w, self.l, self.a, self.b)

    def __repr__(self):
        return self.__str__()


class SLICProcessor(object):
    @staticmethod
    def open_image(path):
        """
        返回:
            三维数组, 行列[LAB]
        """
        # 读取图像并转换为LAB颜色空间
        rgb = io.imread(path)
        rgb_resized = transform.resize(rgb, (224, 224),  mode='reflect')
        lab_arr = color.rgb2lab(rgb_resized)
        return lab_arr

    @staticmethod
    def save_lab_image(path, lab_arr):
        """
        将数组转换为RGB，并保存图像
        :param path:
        :param lab_arr:
        :return:
        """
        # 将LAB颜色空间转换为RGB，并保存图像
        rgb_arr = color.lab2rgb(lab_arr)
        io.imsave(path, rgb_arr)

    def make_cluster(self, h, w):
        h = int(h)
        w = int(w)
        return Cluster(h, w,
                       self.data[h][w][0],
                       self.data[h][w][1],
                       self.data[h][w][2])

    def __init__(self, filename, K, M, fn):
        # 初始化SLICProcessor类
        self.K = K
        self.M = M
        self.fn = fn

        self.data = self.open_image(filename)
        self.image_height = self.data.shape[0]
        self.image_width = self.data.shape[1]
        self.N = self.image_height * self.image_width
        self.S = int(math.sqrt(self.N / self.K))

        self.clusters = []
        self.label = {}
        self.dis = np.full((self.image_height, self.image_width), np.inf)

    def init_clusters(self):
        h = self.S / 2
        w = self.S / 2
        while h < self.image_height:
            while w < self.image_width:
                # 初始化聚类簇的位置并添加到聚类簇列表中
                self.clusters.append(self.make_cluster(h, w))
                w += self.S
            w = self.S / 2
            h += self.S
    
    def reset_pixels(self):
        Cluster.cluster_index = 1
        self.clusters = []
        self.label = {}
        self.dis = np.full((self.image_height, self.image_width), np.inf)

    def get_gradient(self, h, w):
        if w + 1 >= self.image_width:
            w = self.image_width - 2
        if h + 1 >= self.image_height:
            h = self.image_height - 2

        gradient = self.data[h + 1][w + 1][0] - self.data[h][w][0] + \
                   self.data[h + 1][w + 1][1] - self.data[h][w][1] + \
                   self.data[h + 1][w + 1][2] - self.data[h][w][2]
        return gradient

    def move_clusters(self):
        for cluster in self.clusters:
            # 移动聚类簇的位置
            cluster_gradient = self.get_gradient(cluster.h, cluster.w)
            for dh in range(-1, 2):
                for dw in range(-1, 2):
                    _h = cluster.h + dh
                    _w = cluster.w + dw
                    new_gradient = self.get_gradient(_h, _w)
                    if new_gradient < cluster_gradient:
                        cluster.update(_h, _w, self.data[_h][_w][0], self.data[_h][_w][1], self.data[_h][_w][2])
                        cluster_gradient = new_gradient

    def assignment(self):
        for cluster in self.clusters:
            for h in range(cluster.h - 2 * self.S, cluster.h + 2 * self.S):
                if h < 0 or h >= self.image_height: continue
                for w in range(cluster.w - 2 * self.S, cluster.w + 2 * self.S):
                    if w < 0 or w >= self.image_width: continue
                    L, A, B = self.data[h][w]
                    Dc = math.sqrt(
                        math.pow(L - cluster.l, 2) +
                        math.pow(A - cluster.a, 2) +
                        math.pow(B - cluster.b, 2))
                    Ds = math.sqrt(
                        math.pow(h - cluster.h, 2) +
                        math.pow(w - cluster.w, 2))
                    D = math.sqrt(math.pow(Dc / self.M, 2) + math.pow(Ds / self.S, 2))
                    if D < self.dis[h][w]:
                        if (h, w) not in self.label:
                            self.label[(h, w)] = cluster
                            cluster.pixels.append((h, w))
                        else:
                            self.label[(h, w)].pixels.remove((h, w))
                            self.label[(h, w)] = cluster
                            cluster.pixels.append((h, w))
                        self.dis[h][w] = D

    def update_cluster(self):
        for cluster in self.clusters:
            sum_h = sum_w = number = 0
            for p in cluster.pixels:
                sum_h += p[0]
                sum_w += p[1]
                number += 1
            if number != 0:
                _h = int(sum_h / number)
                _w = int(sum_w / number)
                # 更新聚类簇的位置和颜色信息
                cluster.update(_h, _w, self.data[_h][_w][0], self.data[_h][_w][1], self.data[_h][_w][2])

    def save_current_image(self, name):
        # 将LAB图像转换为RGB
        rgb_arr = color.lab2rgb(self.data)
    
        # 创建一个与原图相同大小的空白图像，用于标记边缘
        edge_image = np.zeros_like(rgb_arr)
    
        # 遍历每个聚类簇
        for cluster in self.clusters:
            # 遍历每个像素点
            for p in cluster.pixels:
                # 检查像素点的四个方向是否有不同聚类簇的像素点
                for dh, dw in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    h, w = p[0] + dh, p[1] + dw
                    if 0 <= h < self.image_height and 0 <= w < self.image_width:
                        if (h, w) in self.label and self.label[(h, w)] != cluster:
                            # 如果相邻像素属于不同的聚类簇，则在边缘图像上标记白色
                            edge_image[h][w] = [255, 255, 255]
    
        # 将原图和边缘图像合并，原图不变，边缘白色
        result_image = np.where(edge_image == [255, 255, 255], edge_image, rgb_arr)
    
        # 保存为PNG格式
        io.imsave(name, result_image.astype(np.uint8))

    

    def iterate_10times(self):
        # 迭代十次进行超像素分割
        self.reset_pixels()
        self.init_clusters()
        self.move_clusters()
        for i in trange(10):
            self.assignment()
            self.update_cluster()
            if i == 9:
                name = 'output/test/{k}_{m}/{filen}.png'.format(loop=i, filen=self.fn,k=self.K,m=self.M)
                self.save_current_image(name)
                


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--A", type=int, default=1800, help="Value for K")
    parser.add_argument("--B", type=int, default=100, help="Value for M")
    args = parser.parse_args()

    A = args.A
    B = args.B
    
    #for i in range(1,1001):
    fn=730
    p = SLICProcessor('dataset/images/'+str(fn)+'.png', A, B,fn)
    p.iterate_10times()
    