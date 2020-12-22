import numpy as np
import matplotlib.pyplot as plt

path = 'data/lena.png'
A = plt.imread(path)            # 원본영상 읽기

# plot
plt.figure()                   # figure 생성
plt.imshow(A)   # 이미지 figure에 올리기
plt.show()                     # 보여주기