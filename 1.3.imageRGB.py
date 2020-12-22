import numpy as np
import matplotlib.pyplot as plt

path = 'data/lena.png'
A = plt.imread(path)            # 원본영상
A_gray = (A[:,:,0] + A[:,:,1] + A[:,:,2])/3 # 흑백영상
A_r = np.zeros((512,512,3),np.float32)  # r channel
A_g = np.zeros((512,512,3),np.float32)  # g channel
A_b = np.zeros((512,512,3),np.float32)  # b channel
A_r[:,:,0] = A[:,:,0]
A_g[:,:,1] = A[:,:,1]
A_b[:,:,2] = A[:,:,2]

# plot
fig = plt.figure()
a1 = fig.add_subplot(2,3,1)
a1.imshow(A)
a1 = fig.add_subplot(2,3,2)
a1.imshow(A_gray,cmap='gray')
a4 = fig.add_subplot(2,3,4)
a4.imshow(A_r)
a5 = fig.add_subplot(2,3,5)
a5.imshow(A_g)
a6 = fig.add_subplot(2,3,6)
a6.imshow(A_b)
plt.show()