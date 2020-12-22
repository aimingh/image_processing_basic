import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2

def rgb2gray(img):
    # https://gammabeta.tistory.com/391 참고
    # gray = (img[:,:,0] + img[:,:,1] + img[:,:,2])/3
    # 흑백 이미지로 만들 때 그냥 평균을 써도 된다.
    # 하지만 대부분의 알고리즘은 r,g,b 값에 각각 가중치를 줍니다.
    # 사람의 눈은 동일한 값을 가질 때 녹색이 가장 밝게 보이고 
    # R, B 순으로 밝게 보이기 때문입니다.
    if len(img.shape)==3:
        if img.dtype == np.uint8:
            img = np.float32(img)
            gray = (0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2])
            return np.uint8(gray)
        else:
            return 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]
    else:
        return (img)

def gray2rgb(img):
    if len(img.shape)==2:
        rgb = np.stack((img,)*3,2)
        return rgb
    else:
        return img

def main():
    path = 'data/lena.png'
    img = plt.imread(path)            # 원본영상

    img_gray = rgb2gray(img)                                               # 흑백 영상으로 변환    (512,512,3)->(512,512)
    img_gray_with_cv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray_diff = np.uint8(np.abs(np.int8(img_gray) - np.int8(img_gray_with_cv))) # opencv의 방법과 직접 처리한 방법의 차이를 보기 위한 차영상

    img_gray2rgb = gray2rgb(img_gray)                                          # 변환된 흑백 영상을 3차원으로 컬라 스페이스에 맞도록 변환 
    img_gray2rgb_with_cv = cv2.cvtColor(img_gray_with_cv, cv2.COLOR_GRAY2BGR)          # 컬러영상과 concat 등을 하기 위해서 차원을 같도록 하기 위해 필요 
    img_hconcat = np.hstack((img,img_gray2rgb,img_gray2rgb_with_cv))
    # img_hconcat = np.hstack((img,img_gray))   # 이와같이 흑백과 컬러 영상을 concat 하면 에러가 발생한다.

    # plot
    fig = plt.figure()
    a1 = fig.add_subplot(3,3,1)    
    a1.imshow(img)      
    a1.set_title('original image')
    a4 = fig.add_subplot(3,3,4)
    a4.imshow(img_gray, cmap = 'gray')    
    a4.set_title('gray image')
    a5 = fig.add_subplot(3,3,5)
    a5.imshow(img_gray_with_cv, cmap = 'gray')    
    a5.set_title('gray image using OpenCV')
    a6 = fig.add_subplot(3,3,6)
    a6.imshow(img_gray_diff, cmap = 'gray')    
    a6.set_title('difference of gray images')
    a7 = fig.add_subplot(3,3,7)
    a7.imshow(img_gray2rgb)    
    a7.set_title('gray2rgb image')
    a8 = fig.add_subplot(3,3,8)
    a8.imshow(img_gray2rgb_with_cv) 
    a8.set_title('gray2rgb image using OpenCV')
    a9 = fig.add_subplot(3,3,9)
    a9.imshow(img_hconcat)  
    a9.set_title('concat image')
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()