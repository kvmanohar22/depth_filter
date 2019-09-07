import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_error_img(filename, h, w):
    """ u, v, error (in cm)
    """
    data = np.loadtxt(filename, delimiter=',')
    indices = data[:, :2].astype(np.int32)
    img = np.zeros((h, w), dtype=np.float32)
    img[indices[:, 1], indices[:, 0]] = data[:, 2]
    plt.scatter(indices[:, 1], indices[:, 0], c=data[:, 2])
    plt.show()


def plot(filename, label):
    lines = open(filename, 'r').readlines()
    lines = [line.strip() for line in lines]
    lines = [float(line) for line in lines]
    plt.plot(np.arange(1, len(lines)+1, 1), lines, label=label)


def convergence(filename, label, total):
    plot(filename, label)
    locso, labels = plt.yticks()
    locs = locso/total
    locs *= 100
    labels = ['{:.0f}%'.format(loc) for loc in locs]
    n_lines = len(lines)+5
    plt.xticks(np.arange(0, n_lines, 3), np.arange(0, n_lines, 3))
    plt.yticks(locso, labels)
    plt.xlabel('frame number')
    plt.ylabel('%converged points')
    plt.legend()
    plt.show()


def compare_convergence(files, labels, total):
    for (ifile, label) in zip(files, labels):
        plot(ifile, label)
    locso, labels = plt.yticks()
    locs = locso/total
    locs *= 100
    labels = ['{:.0f}%'.format(loc) for loc in locs]
    plt.yticks(locso, labels)
    plt.xlabel('frame number')
    plt.ylabel('%converged points')
    plt.legend()
    plt.show()

