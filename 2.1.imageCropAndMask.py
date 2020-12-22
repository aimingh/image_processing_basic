import numpy as np
import matplotlib.pyplot as plt

def main():
    path = 'data/lena.png'
    img = plt.imread(path)                                      # 원본영상
    box = [128, 256, 128, 128]                                  # cropping하기위한 박스 위치 x, y, w, h
    mask = np.zeros(img.shape, np.uint8)
    mask[box[1]:box[1]+box[3], box[0]:box[0]+box[2]] = 1        # mask
    img_mask = mask*img                                         # masking된 이미지
    img_crop = img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]  # cropping된 이미지

    # plot
    fig = plt.figure()
    a1 = fig.add_subplot(2,2,1)    
    a1.imshow(img)       # 원본 영상
    a1.set_title('original image')
    a2 = fig.add_subplot(2,2,2)
    a2.imshow(255*mask)    # 마스크 영역, 값이 잘 보이게 하기 위하여 255를 곱
    a2.set_title('mask')
    a3 = fig.add_subplot(2,2,3)
    a3.imshow(img_mask)    # masking된 이미지
    a3.set_title('masking image')
    a4 = fig.add_subplot(2,2,4)
    a4.imshow(img_crop)    # cropping된 이미지
    a4.set_title('cropping image')
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()