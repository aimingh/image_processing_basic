import numpy as np

def bilinear1d(x, y, ratio):
    return (1-ratio)*x + ratio*y

def bilinear2d(point_4, x_ratio, y_ratio):
    return (1-x_ratio)*(1-y_ratio)*point_4[0][0] + x_ratio*(1-y_ratio)*point_4[1][0] + (1-x_ratio)*y_ratio*point_4[0][1] + x_ratio*y_ratio*point_4[1][1] 

def bilinear_interpolation(point_4, x_ratio, y_ratio):
    """
    (0,0), (0,1), (1,0), (1,1)의 값이 있을떄 (x_ratio,y_ratio) 위치의 bilinear interpolation 결과값을 구한다.
    """
    point_4 = np.float32(point_4)

    # 1d bilinear를 이용하여 2d bilinear 결과를 가져올 수 있다.
    x1 = bilinear1d(point_4[0,0], point_4[1,0], x_ratio)
    x2 = bilinear1d(point_4[0,1], point_4[1,1], x_ratio)
    y1 = bilinear1d(x1, x2, y_ratio)

    # 2d bilinear 결과값
    y2 = bilinear2d(point_4, x_ratio, y_ratio)

    print(f'result using 1d bilinear: {y1}')
    print(f'result using 2d bilinear: {y2}')

if __name__ == "__main__":
    point_4 = np.arange(0, 4).reshape((2, 2))
    bilinear_interpolation(point_4, 0.5, 0.5)
