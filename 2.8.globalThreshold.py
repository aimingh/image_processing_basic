import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from cv2 import cv2

def threshold(img, th, max):
    result1 = np.zeros(img.shape, img.dtype)   # THRESH_BINARY
    result2 = np.zeros(img.shape, img.dtype)   # THRESH_BINARY_INV
    result3 = img.copy()                       # THRESH_TRUNC
    result4 = img.copy()                       # THRESH_TOZERO
    result5 = img.copy()                       # THRESH_TOZERO_INV

    result1[np.where(img>th)] = max     # if 
    result2[np.where(img<=th)] = max
    result3[np.where(img>th)] = th
    result4[np.where(img<=th)] = 0
    result5[np.where(img>th)] = 0

    return result1, result2, result3, result4, result5
    
def main():
    img = cv2.imread('data/radial_gradient.jpg',cv2.IMREAD_GRAYSCALE)

    ret,thresh1_cv = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    ret,thresh2_cv = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV) 
    ret,thresh3_cv = cv2.threshold(img,127,255,cv2.THRESH_TRUNC) 
    ret,thresh4_cv = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
    ret,thresh5_cv = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV) 
    thresh1, thresh2, thresh3, thresh4, thresh5 = threshold(img, 127, 255)

    titles = ['Original Image', None,'BINARY','BINARY_CV','BINARY_INV','BINARY_INV_CV','TRUNC','TRUNC_CV','TOZERO', 'TOZERO_CV', 'TOZERO_INV', 'TOZERO_INV_CV']
    images = [img, None, thresh1, thresh1_cv, thresh2, thresh2_cv, thresh3, thresh3_cv, thresh4, thresh4_cv, thresh5, thresh5_cv]
    for i in range(12):
        if titles[i] != None:
            plt.subplot(3,4,i+1)
            plt.imshow(images[i],'gray',vmin=0,vmax=255)
            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()