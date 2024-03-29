import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
plt.style.use('ggplot')

imgfile = '../../../datasets/rpg_synthetic_city/img/img0600_0.png'
img = cv2.imread(imgfile)

VISUALIZE=False
UPPER_BOUND=200


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

# Major error causing pixels
def test_all_error(root):
    files = os.listdir(root)
    files = [os.path.join(root, f) for f in files]
    d_hat = np.zeros(len(files))
    error = np.zeros(len(files))
    for idx, f in enumerate(files):
        uv, t, d = test_single(f)
        e = abs(t-d)
      
        # discard the highly unlikely ones
        if e > UPPER_BOUND:
            continue

        d_hat[idx] = t
        error[idx] = e
 
        # Visualize what's causing the error
        if e > 40:
            print(f, uv, e)
            if VISUALIZE: 
                cv2.circle(img, (int(uv[0]), int(uv[1])), 5, (255, 0, 0))
                cv2.imshow('img', img)
                cv2.waitKey(0)
    plt.scatter(d_hat, error, s=5)
    plt.xlabel('true depth [m]')
    plt.ylabel('error [m]')
    plt.show()

# Pixels where GT was large and yet converged with minimal error
def test_all_converged(root):
    files = os.listdir(root)
    files = [os.path.join(root, f) for f in files]
    d_hat = np.zeros(len(files))
    error = np.zeros(len(files))
    for idx, f in enumerate(files):
        uv, t, d = test_single(f)
        e = abs(t-d)
        
        # discard the highly unlikely ones
        if e > UPPER_BOUND:
            continue

        d_hat[idx] = t
        error[idx] = e

        if t > 40:
            print(f, uv, t, e)
            if VISUALIZE: 
                cv2.circle(img, (int(uv[0]), int(uv[1])), 5, (255, 0, 0))
                cv2.imshow('img', img)
                cv2.waitKey(0)
    plt.scatter(d_hat, error, s=5)
    plt.xlabel('true depth [m]')
    plt.ylabel('error [m]')
    plt.show()


if __name__ == '__main__':
    # test_all_error('./svo/svo_img_600/logs/')
    # test_all_error('./svo/svo_img_1/logs/')
    # test_all_error('./svo/svo_downward/logs/')
    # test_all_error('./svo/svo_downward/logs/')
    # test_all_error('./svo/svo_img_1170/logs/')
    test_all_error('./svo/svo_img_1350/logs/')

