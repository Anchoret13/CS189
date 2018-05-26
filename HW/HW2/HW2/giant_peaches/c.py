#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio


# There is numpy.linalg.lstsq, whicn you should use outside of this classs
def lstsq(A, b):
    return np.linalg.solve(A.T @ A, A.T @ b)


def main():
    data = spio.loadmat('1D_poly.mat', squeeze_me=True)
    x_train = np.array(data['x_train'])
    y_train = np.array(data['y_train']).T
    y_fresh = np.array(data['y_fresh']).T

    n = 20  # max degree
    err_train = np.zeros(n - 1)
    err_fresh = np.zeros(n - 1)

    # fill in err_fresh and err_train
    # YOUR CODE HERE
    for i in range(n-1):
        D = i+1
        for k in range(D+1):
            if (k == 0):
                Xf = np.array([1] * n)
            else:
                Xf = np.vstack([np.power(x_train, k), Xf])
        Xf = Xf.T

        w = lstsq(Xf,y_train)
        #y_predict = Xf@w
        #err[i] = (np.linalg.norm(y_predict-y_train)**2)/n
        err_train[i] = (np.linalg.norm(y_train-y_train)**2)/n
        err_fresh[i] = (np.linalg.norm(y_fresh-y_train)**2)/n


    plt.figure()
    plt.ylim([0, 6])
    plt.plot(err_train, label='train')
    plt.plot(err_fresh, label='fresh')
    plt.legend()
    plt.show()
##WEIRD ANS
if __name__ == "__main__":
    main()
