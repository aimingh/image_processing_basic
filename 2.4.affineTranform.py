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

def affineTransform(gray, affineMatrix):
    h, w = gray.shape
    result = np.zeros((h,w),gray.dtype)

    for x in range(h):                          # 모든 좌표에 대하여 affinematrix를 행렬곱해준다.
        for y in range(w):  
            yx = np.array([y, x, 1])
            ydot, xdot, _ = np.dot(affineMatrix, yx)
            if round(ydot)<h and round(xdot)<w:
                result[int(round(ydot)), int(round(xdot))] = gray[y,x]

    return result

if __name__ == "__main__":
    path = 'data/lena.png'
    img = plt.imread(path)
    gray = rgb2gray(img)
    scaling1 = np.array([[2,0,0],
                         [0,1,0],
                         [0,0,1]])

    result1 = affineTransform(gray, scaling1)

    scaling2 = np.array([[0.5,0,0],
                          [0,1,0],
                          [0,0,1]])

    result2 = affineTransform(gray, scaling2)

    translation1 = np.array([[1,0,256],
                             [0,1,0],
                             [0,0,1]])

    result3 = affineTransform(gray, translation1)

    translation2 = np.array([[1,0,0],
                             [0,1,300],
                             [0,0,1]])

    result4 = affineTransform(gray, translation2)

    translation3 = np.array([[1,0,256],
                             [0,1,300],
                             [0,0,1]])

    result5 = affineTransform(gray, translation3)

    # plot
    fig = plt.figure()
    a1 = fig.add_subplot(2,3,1)    
    a1.imshow(gray, cmap = 'gray')      
    a1.set_title('original image')
    a2 = fig.add_subplot(2,3,2)
    a2.imshow(result1, cmap = 'gray')    
    a2.set_title('scaling1')
    a3 = fig.add_subplot(2,3,3)
    a3.imshow(result2, cmap = 'gray')    
    a3.set_title('scaling2')
    a4 = fig.add_subplot(2,3,4)
    a4.imshow(result3, cmap = 'gray')    
    a4.set_title('Translate1')
    a5 = fig.add_subplot(2,3,5)
    a5.imshow(result4, cmap = 'gray')    
    a5.set_title('Translate2')
    a6 = fig.add_subplot(2,3,6)
    a6.imshow(result5, cmap = 'gray')    
    a6.set_title('Translate3')
    fig.tight_layout()
    plt.show()

