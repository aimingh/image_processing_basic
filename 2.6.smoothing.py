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

def gkern(l=5, sig=1.):
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    return kernel / np.sum(kernel)

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
    
    kernel = gkern(11, 3)

    img_blur = filtering(img, kernel)
    img_blur_with_cv = cv2.GaussianBlur(rgb2gray(img),(11,11),3)

    fig = plt.figure()
    a1 = fig.add_subplot(2,2,1)    
    a1.imshow(rgb2gray(img), cmap = 'gray')       
    a1.set_title('gray image')
    a2 = fig.add_subplot(2,2,3)    
    a2.imshow(img_blur, cmap = 'gray') 
    a2.set_title('Blur') 
    a3 = fig.add_subplot(2,2,4)    
    a3.imshow(img_blur_with_cv, cmap = 'gray')  
    a3.set_title('Blur using OpenCV')
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
