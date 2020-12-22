import numpy as np
import matplotlib.pyplot as plt

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

def histogram(img):
    hist = [len(img[np.where(img==n)]) for n in range(256)]
    return np.array(hist)   # 히스토그램 각 밝기값에 따른 픽셀 수 

def cumsum(h):
	return [sum(h[:i+1]) for i in range(len(h))]

def histogramEq(img):
    h, w = img.shape
    hist = histogram(img)/(h*w)
    cdf = np.array(cumsum(hist)) # cdf 생성
    cdf_norm = np.uint8(255 * cdf) # normalization 0~255
    result = np.zeros_like(img)
    for i in range(0, h):
    	for j in range(0, w):
    		result[i, j] = cdf_norm[img[i, j]]  # normalization된 cdf를 이용하여 맵핑
    return np.uint8(result)

if __name__ == "__main__":
    path = 'data/lena.png'
    img = plt.imread(path)
    if img.dtype!=np.uint8:
        img = (255*img)
    else:
        np.float32(img)
    gray = rgb2gray(img)

    dark_img = np.uint8(127*(gray-gray.min())/(gray.max()-gray.min()))    # 어둡고 히스토그램이 쏠려있는 이미지를 만들어준다.
    hist1 = histogram(dark_img)

    result = histogramEq(dark_img)  # 히스토그램 평활화
    hist2 = histogram(result)

    # plot
    fig = plt.figure()
    a1 = fig.add_subplot(2,2,1)    
    a1.imshow(np.stack((dark_img,)*3,2))      
    a1.set_title('darken image')
    a2 = fig.add_subplot(2,2,2)
    a2.bar(np.arange(256), hist1)    
    a2.set_title('histogram of darken image')
    a3 = fig.add_subplot(2,2,3)    
    a3.imshow(np.stack((result,)*3,2))      
    a3.set_title('result image')
    a4 = fig.add_subplot(2,2,4)
    a4.bar(np.arange(256), hist2)    
    a4.set_title('histogram of result image')
    fig.tight_layout()
    plt.show()