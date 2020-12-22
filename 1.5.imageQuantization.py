import numpy as np
import matplotlib.pyplot as plt

depth_level = 8    # N level로 밝기 단계를 줄인다.

path = 'data/lena.png'
A = plt.imread(path)            # 원본영상
A_gray = (A[:,:,0] + A[:,:,1] + A[:,:,2])/3
    
A_ = np.round((depth_level-1)*A)/(depth_level-1)    # N level로 영상을 바꾼다. 양자화 Re-Quantization
A_gray_ = np.round((depth_level-1)*A_gray)/(depth_level-1)

# plot
fig = plt.figure()
a1 = fig.add_subplot(2,2,1)
a1.imshow(A_gray, cmap='gray')
a2 = fig.add_subplot(2,2,2)
a2.imshow(A_gray_, cmap='gray')
a3 = fig.add_subplot(2,2,3)
a3.imshow(A)
a4 = fig.add_subplot(2,2,4)
a4.imshow(A_)
plt.show()