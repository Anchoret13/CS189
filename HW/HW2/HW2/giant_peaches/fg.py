import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as spio

data = spio.loadmat('polynomial_regression_samples.mat', squeeze_me=True)
data_x = data['x']
data_y = data['y']
Kc = 4  # 4-fold cross validation
KD = 6  # max D = 6
LAMBDA = [0, 0.05, 0.1, 0.15, 0.2]

def lstsq(A, b,lambda_=0):
    #return np.linalg.solve(A.T @ A, A.T @ b)
    return np.linalg.solve(A.T @ A + lambda_ * np.eye(A.shape[1]), A.T @b)
def Deal(x,D):
    n_feature = x.shape[1] 
    Q = [(np.ones(x.shape[0]), 0, 0)] 
    i = 0 
    while Q[i][1] < D: 
        cx, degree, last_index = Q[i] 
        for j in range(last_index, n_feature): 
            Q.append((cx * x[:, j], degree + 1, j)) 
        i += 1 
    return np.column_stack([q[0] for q in Q])
    
def fit(D, lambda_):
    # YOUR CODE TO COMPUTE THE AVERAGE ERROR PER SAMPLE
    #training
    NT = int(data_x.shape[0]*(Kc - 1)/Kc)
    #validation
    NV = int(NT/(Kc-1))
    Etrain = np.zeros(4)
    Evalid = np.zeros(4)
    for c in range(4):
        valid_x = data_x[c*NV:(c+1)*NV]
        valid_y = data_y[c*NV:(c+1)*NV]
        train_x = np.delete(data_x,list(range(c*NV,(c+1)*NV)),axis = 0)
        train_y = np.delete(data_x,list(range(c*NV,(c+1)*NV)))
    #pass
        w = lstsq(Deal(train_x,D),Deal(valid_x,D),lambda_ = lambda_)
        Etrain[c] = np.mean((train_y - Deal(train_x,D) @ w)**2)
        Evalid[c] = np.mean((valid_y - Deal(valid_x,D) @ w)**2)
    return (np.mean(Etrain),np.mean(Evalid))



def main():
    np.set_printoptions(precision=11)
    Etrain = np.zeros((KD, len(LAMBDA)))
    Evalid = np.zeros((KD, len(LAMBDA)))
    for D in range(KD):
        print(D)
        for i in range(len(LAMBDA)):
            Etrain[D, i], Evalid[D, i] = fit(D + 1, LAMBDA[i])

    print('Average train error:', Etrain, sep='\n')
    print('Average valid error:', Evalid, sep='\n')

    # YOUR CODE to find best D and i
    D=np.unravel_index(Evalid.argmin())
    i=np.unravel_index(Evalid.shape)


if __name__ == "__main__":
    main()
