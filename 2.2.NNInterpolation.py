import numpy as np
from math import floor

def nn_interpolation(point_4, x_ratio, y_ratio):
    """
    (0,0), (0,1), (1,0), (1,1)의 값이 있을떄 (x_ratio,y_ratio) 위치의 nearest neighbor interpolation 결과값을 구한다.
    """
    result = point_4[floor(x_ratio), floor(y_ratio)]
    print(f'result using 1d bilinear: {result}')

if __name__ == "__main__":
    point_4 = np.arange(0, 4).reshape((2, 2))
    nn_interpolation(point_4, 0.5, 0.5)
