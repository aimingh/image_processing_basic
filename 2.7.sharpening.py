import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from cv2 import cv2

def rgb2gray(img):
    if len(img.shape)==3:
        if img.dtype == np.uint8:
            img = np.float32(img)
            gray = (0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2])
            return np.uint8(gray)
        else:
            return 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]
    else:
        return (img)

def filtering(img, kernel):
    img = rgb2gray(img)
    x, y = img.shape
    kernel_size, _ = kernel.shape
    img_padding = np.zeros((x+kernel_size-1, y+kernel_size-1), img.dtype)
    pad = int(0.5*(kernel_size-1))
    img_padding[pad:-pad,pad:-pad] = img
    img_th = np.zeros(img.shape, img.dtype)

    for i in range(x):
        for j in range(y):
            img_th[i,j] = np.sum(kernel*img_padding[i:i+kernel_size, j:j+kernel_size])

    return img_th

def main():
    img = plt.imread('data/lena.png')

    kernel = np.array([[-1,-1,-1],
                       [-1,8,-1],
                       [-1,-1,-1],])
    
    img_edge = filtering(img, kernel)
    img_edge = np.abs(img_edge)
    img_edge[np.where(img_edge>1)]=1

    kernel = np.array([[-1,-1,-1],
                       [-1,9,-1],
                       [-1,-1,-1],])
    
    img_sharp = filtering(img, kernel)
    img_sharp = np.abs(img_sharp)
    img_sharp[np.where(img_sharp>1)]=1

    fig = plt.figure()
    a1 = fig.add_subplot(2,2,1)    
    a1.imshow(rgb2gray(img), cmap = 'gray')       
    a1.set_title('gray image')
    a2 = fig.add_subplot(2,2,2)    
    a2.imshow(img_edge, cmap = 'gray') 
    a2.set_title('edge image') 
    a3 = fig.add_subplot(2,2,3)    
    a3.imshow(img_sharp, cmap = 'gray') 
    a3.set_title('sharpened image') 
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
