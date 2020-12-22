import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from cv2 import cv2

def rgb2gray(img):
    if len(img.shape)==3:
        img = np.float32(img)
        gray = (0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2])
        return np.uint8(gray)
    else:
        return (img)

def gkern(l=5, sig=1.):
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    return kernel / np.sum(kernel)

def adaptiveGaussianThreshold(img, kernel_size = 11, kernel_sigma = 3, c = 0):
    img = rgb2gray(img)
    x, y = img.shape
    img_padding = np.zeros((x+kernel_size-1, y+kernel_size-1), np.uint8)
    pad = int(0.5*(kernel_size-1))
    img_padding[pad:-pad,pad:-pad] = img
    img_th = np.zeros(img.shape, np.uint8)
    kernel = gkern(kernel_size, kernel_sigma)

    for i in range(x):
        for j in range(y):
            th = np.sum(kernel*img_padding[i:i+kernel_size, j:j+kernel_size])
            if img[i,j] > th - c:
                img_th[i,j] = 255

    return img_th
    
def main():
    img = plt.imread('data/sudoku.jpg')

    img_th = adaptiveGaussianThreshold(img, 11, 3, 4)
    img_th2 = cv2.adaptiveThreshold(rgb2gray(img),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,4)

    fig = plt.figure()
    a1 = fig.add_subplot(1,3,1)    
    a1.imshow(img)       
    a1.set_title('original image')
    a2 = fig.add_subplot(1,3,2)    
    a2.imshow(img_th, cmap = 'gray') 
    a2.set_title('adap th') 
    a3 = fig.add_subplot(1,3,3)    
    a3.imshow(img_th2, cmap = 'gray')  
    a3.set_title('adap th using OpenCV')
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
