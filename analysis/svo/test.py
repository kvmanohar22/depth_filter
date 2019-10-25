import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
plt.style.use('ggplot')

imgfile = '../../../../../datasets/rpg_synthetic_city/img/img0600_0.png'
img = cv2.imread(imgfile)

def parse(lines):
    lines = [line.strip() for line in lines]
    lines = [line.split(' ') for line in lines]
    arr = np.zeros((len(lines), 2))
    for idx, line in enumerate(lines):
        arr[idx] = np.array([float(i) for i in line])
    return arr

def test_single(path):
    data = open(path).readlines()
    data = parse(data)
    return data[0], data[1, 0], data[-1, 0]

def test_all():
    files = os.listdir('svo')
    files = [os.path.join('svo', f) for f in files]
    d_hat = np.zeros(len(files))
    error = np.zeros(len(files))
    for idx, f in enumerate(files):
        uv, t, d = test_single(f)
        e = abs(t-d)
        d_hat[idx] = t
        error[idx] = e
        if e > 40:
            print(f, uv, e)
            cv2.circle(img, (int(uv[0]), int(uv[1])), 5, (255, 0, 0))
            cv2.imshow('img', img)
            cv2.waitKey(0)
    plt.scatter(d_hat, error, s=5)
    plt.xlabel('true depth [m]')
    plt.ylabel('error [m]')
    plt.show()

if __name__ == '__main__':
    test_all()

