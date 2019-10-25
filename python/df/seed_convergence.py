#!/usr/bin/python

import cv2
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.style.use('ggplot')

def plot_error_img(filename, h, w, outfile):
    """ u, v, error (in m)
    """
    data = np.loadtxt(filename, delimiter=',')
    indices = data[:, :2].astype(np.int32)
    img = np.zeros((h, w), dtype=np.float32)
    min_error = np.min(data[:, 2])
    error = data[:, 2]*100
    img[indices[:, 1], indices[:, 0]] = error
    img = np.flip(img, axis=0)

    # mask the invalid regions
    img = ma.masked_where(img <= 1e-12, img)

    x,y=np.meshgrid(np.arange(0, w, 1), np.arange(0, h, 1))
    pcm = plt.pcolormesh(x, y, img, norm=LogNorm(vmin=1e-2, vmax=np.max(img)), cmap='RdBu_r')
    plt.colorbar(pcm)

    plt.savefig(outfile)
    # plt.show()
    plt.cla()
    plt.clf()

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
    # n_lines = len(lines)+5
    # plt.xticks(np.arange(0, n_lines, 3), np.arange(0, n_lines, 3))
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

def compare_mean_change(ifile):
    data = np.loadtxt(ifile)
    diff_predicted = abs(data[:,1]-data[:,0])
    diff_truth = abs(data[:,2]-data[:,0])
    valid_idxs = diff_predicted < 100
    diff_predicted = diff_predicted[valid_idxs]
    
    # difference between predicted and initialized depth
    plt.hist(diff_truth, bins=300, density=True, label='true density')
    plt.hist(diff_predicted, bins=300, density=True, label='predicted density', alpha=0.5)
    plt.xlabel('absolute change')
    plt.ylabel('% of converged seeds [normalized]')
    plt.legend() 
    plt.title('initialized depth={:.2f}'.format(data[0, 0]))
    plt.show()
 
def mean_change(ifile):
    data = np.loadtxt(ifile)
    diff_predicted = abs(data[:,1]-data[:,0])
    diff_truth = abs(data[:,2]-data[:,0])
    valid_idxs = data[:, 2] < 200
    diff_truth = diff_truth[valid_idxs]
    
    # difference between predicted and initialized depth
    plt.hist(diff_truth, bins=1000, density=True)
    plt.xlabel('|true_depth - initialized_depth|')
    plt.ylabel('% of converged seeds')
    plt.title('initialized depth={:.2f}'.format(data[0, 0]))
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    # plt.show()
   
    # difference between predicted and initialized depth
    plt.cla()
    plt.clf()
    valid_idxs = diff_predicted < 200
    diff_predicted = diff_predicted[valid_idxs]
    plt.hist(diff_predicted, bins=1000, density=True)
    plt.xlabel('|predicted_depth - initialized_depth|')
    plt.ylabel('% of converged seeds')
    plt.title('initialized depth={:.2f}'.format(data[0, 0]))
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    # plt.show()

    # distribution of depth 
    plt.cla()
    plt.clf()
    plt.hist(data[:, 1], bins=1000)
    locso, labels = plt.yticks()
    locs = locso/len(data)
    locs *= 100
    labels = ['{:.0f}%'.format(loc) for loc in locs]
    plt.yticks(locso, labels)
    plt.show()


if __name__ == '__main__':
    # plot_error_img('../../analysis/forward/log5/depth_filter_rpg_synthetic_forward.txt', 480, 640, 'forward.pdf')

    logfile = '../../analysis/downward/log3/depth_filter_sin2_tex2_h1_v8_d.txt'
    plot_error_img(logfile, 480, 752, logfile.replace('.txt', '.pdf'))
    # plot_error_img('../../analysis/forward/log7/depth_filter_rpg_synthetic_forward.txt', 480, 640, 'downward.svg')
    # plot_error_img('../../analysis/forward/log8/depth_filter_rpg_synthetic_forward.txt', 480, 640, 'downward.svg')
    # plot_error_img('../../analysis/forward/log9/depth_filter_rpg_synthetic_forward.txt', 480, 640, 'downward.svg')
    logfile = '../../analysis/forward/log10/depth_filter_rpg_synthetic_forward.txt'
    plot_error_img(logfile, 480, 640, logfile.replace('.txt', '.pdf'))
    # logfile = '../../analysis/forward/log11/depth_filter_rpg_synthetic_forward.txt'
    # plot_error_img(logfile, 480, 640, logfile.replace('.txt', '.pdf'))
    # logfile = '../../analysis/forward/log12/depth_filter_rpg_synthetic_forward.txt'
    # plot_error_img(logfile, 480, 640, logfile.replace('.txt', '.pdf'))

    # mean_change('../../analysis/forward/log5/depth_filter_rpg_synthetic_forward_mean_change.txt')
    # mean_change('../../analysis/forward/log10/depth_filter_rpg_synthetic_forward_mean_change.txt')
    # mean_change('../../analysis/downward/log3/depth_filter_sin2_tex2_h1_v8_d_mean_change.txt')

