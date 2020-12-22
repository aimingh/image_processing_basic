import numpy as np
import matplotlib.pyplot as plt

A = np.array([[0,0,0,0,0,0,0,0,0,0],    # 원본영상, 숫자2
              [0,0,0,1,1,1,1,0,0,0],    # 행렬로 영상을 표현 가능하다.
              [0,0,0,1,1,1,1,0,0,0],
              [0,0,0,1,1,1,1,0,0,0],
              [0,0,0,1,1,1,1,0,0,0],
              [0,0,0,1,1,1,1,0,0,0],
              [0,0,0,1,1,1,1,0,0,0],
              [0,0,0,1,1,1,1,0,0,0],
              [0,0,0,1,1,1,1,0,0,0],
              [0,0,0,0,0,0,0,0,0,0],])

plt.figure()                   # figure 생성
plt.imshow(A, cmap = 'gray')   # 이미지 figure에 올리기
plt.show()                     # 보여주기