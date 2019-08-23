#!/usr/bin/python

import os
import shutil
from natsort import natsorted, ns
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
plt.style.use('ggplot')
 
BASE = "./logs/df_upper_nview_stats"

def analyze_single_file(src, png):
    plt.cla()
    plt.clf()

    data = np.loadtxt(src)
    data = data[2:, :]
    z_true = data[0, 0]
    print('true depth = ', z_true)
    data = data[1:, :]
    valid_idxs = data[:, 1] != -2
    data = data[valid_idxs]
    plt.plot(data[:, 0], data[:, 1], label='score')
   
    try:
        nearest = data[np.argmin(np.square(data[:, 0] - z_true))]
    except ValueError:
        return
    
    plt.scatter(nearest[0], nearest[1], marker='*', color='g', label='true depth')
    plt.xlabel("depth (in m)")
    plt.ylim(-1.0, 1.0)
    plt.ylabel("NCC score")
    plt.title("$\hat d$ = {:.2f}".format(z_true))
    plt.legend()
    plt.savefig(png)

def generate_nview_stats(dir):
    plt.cla()
    plt.clf()

    files = os.listdir(join(dir, 'scores'))
    files = [join(dir, 'scores', file) for file in files]
    for file in files: 
        data = np.loadtxt(file)
        data = data[2:, :]
        z_true = data[0, 0]
        data = data[1:, :]
        if len(data) == 100:
            break
    z_range = data[:, 0]
    z_true_idx = np.argmin(np.square(z_range-z_true))
    aggregated_hist = {}
    aggregated_hist.update(zip(z_range, np.zeros_like(z_range).astype(np.int32))) 
    colors = ['forestgreen' for _ in range(100)]
    colors[z_true_idx] = 'r'
    for file in files:
        data = np.loadtxt(file)
        data = data[2:, :]
        z_true = data[0, 0]
        data = data[1:, :]
        local_maxima_idxs = argrelextrema(data[:, 1], np.greater)
        local_maxima_z = np.squeeze(data[local_maxima_idxs, 0])
        min_dist_idxs = np.argmin(np.square(local_maxima_z[..., None] - z_range), axis=-1)
        min_dist_vals = z_range[min_dist_idxs]
        if isinstance(min_dist_vals, np.float64):
            min_dist_vals = [min_dist_vals]
        for update_i in range(len(min_dist_vals)):
            aggregated_hist[min_dist_vals[update_i]] += 1
    plt.bar(aggregated_hist.keys(), aggregated_hist.values(), color=colors)
    plt.xlabel('d (in meters')
    plt.ylabel('#local maxima')
    plt.title('20 observations $\hat d$ = {:.2f}'.format(z_true))
    plt.legend()
    plt.savefig(dir+'/n-view.svg')

def analyze(dir):
    files = os.listdir(join(dir, 'scores'))
    for file in files:
        srcfile = join(dir, 'scores', file)
        pngfile = srcfile.replace('/scores/', '/plots/')
        pngfile = pngfile.replace('.score', '.svg')
        analyze_single_file(srcfile, pngfile)

def move_files(files):
    split = files[0].split('_')
    dirname = '{}_{}_{}_{}'.format(*split[:4])
    if not os.path.isdir(join(BASE, dirname)):
        os.makedirs(join(BASE, dirname, 'scores'))
        os.makedirs(join(BASE, dirname, 'plots'))
    for file in files:
        srcfile = join(BASE, file)
        shutil.move(srcfile, join(BASE, dirname, 'scores', file))         

if __name__ == '__main__':
    files = natsorted(os.listdir(BASE), key=lambda x: x.lower())
    files = [file for file in files if os.path.isfile(join(BASE, file))]
    n_sets = len(files) // 20
    for idx in range(n_sets):
        move_files(files[idx*20:(idx+1)*20])
    
    dirs = natsorted(os.listdir(BASE), key=lambda x: x.lower())
    dirs = [join(BASE, dir) for dir in dirs]
    for i, dir in enumerate(dirs):
        print('processing {:3d} / {:3d}'.format(i, len(dirs)))
        generate_nview_stats(dir)
