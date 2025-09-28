import math
import numpy as np
from skimage import io, color, transform
from ctypes import cdll, c_double, POINTER, c_int, c_float

def slic_segment(image, K, M, prob, so_path="./SLICSP_v0928.so"):
    
    if isinstance(image, str):
        rgb = io.imread(image)
    else:
        rgb = image
    rgb_resized = transform.resize(rgb, (224, 224), mode='reflect')
    data = color.rgb2lab(rgb_resized)
    image_height, image_width = data.shape[:2]
    N = image_height * image_width
    S = int(math.sqrt(N / K))
    L = data[:,:,0].flatten()
    A = data[:,:,1].flatten()
    B = data[:,:,2].flatten()
    
    labels = np.zeros(image_width * image_height, dtype=np.float64)
    X_mask = np.ones((1, image_height, image_width, 3), dtype=np.float32)
    lib = cdll.LoadLibrary(so_path)
    # 只需传递空的CX, CY, CL, CA, CB数组，C++端会初始化
    CX = np.zeros(K, dtype=np.float64)
    CY = np.zeros(K, dtype=np.float64)
    CL = np.zeros(K, dtype=np.float64)
    CA = np.zeros(K, dtype=np.float64)
    CB = np.zeros(K, dtype=np.float64)
    SeedsNum = len(CX)
    CX_ptr = CX.ctypes.data_as(POINTER(c_double))
    CY_ptr = CY.ctypes.data_as(POINTER(c_double))
    CL_ptr = CL.ctypes.data_as(POINTER(c_double))
    CA_ptr = CA.ctypes.data_as(POINTER(c_double))
    CB_ptr = CB.ctypes.data_as(POINTER(c_double))
    L_ptr = L.ctypes.data_as(POINTER(c_double))
    A_ptr = A.ctypes.data_as(POINTER(c_double))
    B_ptr = B.ctypes.data_as(POINTER(c_double))
    labels_ptr = labels.ctypes.data_as(POINTER(c_double))
    X_mask_ptr = X_mask.ctypes.data_as(POINTER(c_float))

    lib.SLICSP(CX_ptr, CY_ptr, CL_ptr, CA_ptr, CB_ptr,
               c_int(SeedsNum), L_ptr, A_ptr, B_ptr,
               c_int(image_width), c_int(image_height),
               c_double(S), c_double(M), labels_ptr,
    c_double(prob), X_mask_ptr)
    return labels.reshape((image_height, image_width)), SeedsNum, X_mask


    