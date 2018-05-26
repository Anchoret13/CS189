import pickle
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def flat(n):
    nx = n.shape[0]
    row = np.ndarray.flatten(nx).size / nx
    sol = np.zeros((nx,row))
    for i in range(nx):
        sol[i,:] = np.ndarray.flatten(n.x_train[i,:,:,:])
    return sol
class HW3_Sol(object):

    def __init__(self):
        pass

    def load_data(self):
        self.x_train = pickle.load(open('x_train.p','rb'), encoding='latin1')
        self.y_train = pickle.load(open('y_train.p','rb'), encoding='latin1')
        self.x_test = pickle.load(open('x_test.p','rb'), encoding='latin1')
        self.y_test = pickle.load(open('y_test.p','rb'), encoding='latin1')

    def plotflatall(self):
        self.x_train_flat = flat(self.x_train)
        self.x_test_flat = flat(self.x_test)

    
if __name__ == '__main__':

    hw3_sol = HW3_Sol()

    hw3_sol.load_data()

    # Your solution goes here
    
    #Q(a)
    plt.imshow(hw3_sol.x_train[0])
    print("u0={}".format(hw3_sol.y_train[0])) 
    #plt.show()
    plt.imshow(hw3_sol.x_train[10])
    print("u0={}".format(hw3_sol.y_train[10])) 
    #plt.show()
    plt.imshow(hw3_sol.x_train[0])
    print("u0={}".format(hw3_sol.y_train[20])) 
    #plt.show()

    #Q(b)
    #print(hw3_sol.xtrain.shape)
    #X = hw3_sol.x_train.reshape(hw3_sol.x_train.shape[0],-1)
    #U = hw3_sol.x_train
    
