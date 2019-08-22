import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

FILE="./logs/ncc_log.score"

def analyze():
    data = np.loadtxt(FILE)
    valid_idxs = data[:, 1] != -2
    data = data[valid_idxs]
    plt.plot(data[:, 0], data[:, 1])
    plt.xlabel("$\hat d$")
    plt.ylabel("NCC score")
    plt.show()

if __name__ == '__main__':
    analyze()

