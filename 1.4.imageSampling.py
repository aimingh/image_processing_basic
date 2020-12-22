import numpy as np
import matplotlib.pyplot as plt

sampling_level = 8    # 샘플링 수를 감소시키는 비율

path = 'data/lena.png'
A = plt.imread(path)            # 원본영상
A_gray = (A[:,:,0] + A[:,:,1] + A[:,:,2])/3
A_ = A[::sampling_level, ::sampling_level]
A_gray_ = A_gray[::sampling_level, ::sampling_level]

# plot
fig = plt.figure()
a1 = fig.add_subplot(2,2,1)
a1.imshow(A)
a2 = fig.add_subplot(2,2,2)
a2.imshow(A_)
a3 = fig.add_subplot(2,2,3)
a3.imshow(A_gray,cmap='gray')
a4 = fig.add_subplot(2,2,4)
a4.imshow(A_gray_,cmap='gray')
plt.show()