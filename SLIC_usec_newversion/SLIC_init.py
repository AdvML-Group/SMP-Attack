import math
import numpy as np
from skimage import io, color, transform
from ctypes import cdll, c_double, POINTER, c_int

class Cluster:
    def __init__(self, h, w, l=0, a=0, b=0):
        self.h = h
        self.w = w
        self.l = l
        self.a = a
        self.b = b

class SLICInit:
    def __init__(self, image_path, K, M):
        rgb = io.imread(image_path)
        rgb_resized = transform.resize(rgb, (224, 224), mode='reflect')
        self.data = color.rgb2lab(rgb_resized)
        self.image_height, self.image_width = self.data.shape[:2]
        self.K = K
        self.M = M
        self.N = self.image_height * self.image_width
        self.S = int(math.sqrt(self.N / self.K))
        self.clusters = []

    def get_gradient(self, h, w):
        h = min(h, self.image_height - 2)
        w = min(w, self.image_width - 2)
        gradient = (
            self.data[h + 1][w + 1][0] - self.data[h][w][0] +
            self.data[h + 1][w + 1][1] - self.data[h][w][1] +
            self.data[h + 1][w + 1][2] - self.data[h][w][2]
        )
        return gradient

    def init_clusters(self):
        h = self.S // 2
        w = self.S // 2
        while h < self.image_height:
            while w < self.image_width:
                self.clusters.append(Cluster(h, w, *self.data[h][w]))
                w += self.S
            w = self.S // 2
            h += self.S

    def move_clusters(self):
        for cluster in self.clusters:
            min_grad = self.get_gradient(cluster.h, cluster.w)
            min_h, min_w = cluster.h, cluster.w
            for dh in range(-1, 2):
                for dw in range(-1, 2):
                    _h, _w = cluster.h + dh, cluster.w + dw
                    grad = self.get_gradient(_h, _w)
                    if grad < min_grad:
                        min_grad = grad
                        min_h, min_w = _h, _w
            cluster.h, cluster.w = min_h, min_w
            cluster.l, cluster.a, cluster.b = self.data[min_h][min_w]

    def get_centers(self):
        CX = np.array([c.w for c in self.clusters], dtype=np.float64)
        CY = np.array([c.h for c in self.clusters], dtype=np.float64)
        CL = np.array([c.l for c in self.clusters], dtype=np.float64)
        CA = np.array([c.a for c in self.clusters], dtype=np.float64)
        CB = np.array([c.b for c in self.clusters], dtype=np.float64)
        return CX, CY, CL, CA, CB


def slic_segment(image, K, M, so_path="./libSLICSP.so"):
    # 支持 image 为路径或数组
    if isinstance(image, str):
        rgb = io.imread(image)
    else:
        rgb = image
    rgb_resized = transform.resize(rgb, (224, 224), mode='reflect')
    data = color.rgb2lab(rgb_resized)
    image_height, image_width = data.shape[:2]
    N = image_height * image_width
    S = int(math.sqrt(N / K))
    # 初始化中心点
    clusters = []
    h = S // 2
    w = S // 2
    while h < image_height:
        while w < image_width:
            l, a, b = data[h][w]
            clusters.append((h, w, l, a, b))
            w += S
        w = S // 2
        h += S
    # move_clusters逻辑：让中心点避开高梯度区域
    def get_gradient(data, h, w):
        h = min(h, data.shape[0] - 2)
        w = min(w, data.shape[1] - 2)
        return (data[h+1][w+1][0] - data[h][w][0] +
                data[h+1][w+1][1] - data[h][w][1] +
                data[h+1][w+1][2] - data[h][w][2])
    for idx, (h, w, l, a, b) in enumerate(clusters):
        min_grad = get_gradient(data, h, w)
        min_h, min_w = h, w
        for dh in range(-1, 2):
            for dw in range(-1, 2):
                _h, _w = h + dh, w + dw
                if 0 <= _h < image_height and 0 <= _w < image_width:
                    grad = get_gradient(data, _h, _w)
                    if grad < min_grad:
                        min_grad = grad
                        min_h, min_w = _h, _w
        clusters[idx] = (min_h, min_w, *data[min_h][min_w])
    CX = np.array([c[1] for c in clusters], dtype=np.float64)
    CY = np.array([c[0] for c in clusters], dtype=np.float64)
    CL = np.array([c[2] for c in clusters], dtype=np.float64)
    CA = np.array([c[3] for c in clusters], dtype=np.float64)
    CB = np.array([c[4] for c in clusters], dtype=np.float64)
    SeedsNum = len(CX)
    L = data[:,:,0].flatten()
    A = data[:,:,1].flatten()
    B = data[:,:,2].flatten()
    labels = np.zeros(image_width * image_height, dtype=np.float64)
    lib = cdll.LoadLibrary(so_path)
    CX_ptr = CX.ctypes.data_as(POINTER(c_double))
    CY_ptr = CY.ctypes.data_as(POINTER(c_double))
    CL_ptr = CL.ctypes.data_as(POINTER(c_double))
    CA_ptr = CA.ctypes.data_as(POINTER(c_double))
    CB_ptr = CB.ctypes.data_as(POINTER(c_double))
    L_ptr = L.ctypes.data_as(POINTER(c_double))
    A_ptr = A.ctypes.data_as(POINTER(c_double))
    B_ptr = B.ctypes.data_as(POINTER(c_double))
    labels_ptr = labels.ctypes.data_as(POINTER(c_double))
    lib.SLICSP(CX_ptr, CY_ptr, CL_ptr, CA_ptr, CB_ptr,
               c_int(SeedsNum), L_ptr, A_ptr, B_ptr,
               c_int(image_width), c_int(image_height),
               c_double(S), c_double(M), labels_ptr)
    return labels.reshape((image_height, image_width)), SeedsNum
