import matplotlib
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

def generate_data(n):
    """
    This function generates data of size n.
    """
    #TODO implement this
    x = np.random.randn(n,2) * np.sqrt(5)
    z = np.random.randn(n)
    y = np.sum(x,axis=1)*z
    return (x,y)

def tikhonov_regression(X,Y,Sigma):
    """
    This function computes w based on the formula of tikhonov_regression.
    """
    #TODO implement this
    #var = np.linalg.inv((X.T.dot(X)+np.linalg.inv(Sigma)))
    w = np.linalg.inv((X.T.dot(X)+np.linalg.inv(Sigma))).dot(X.T.dot(Y))
    return w

def compute_mean_var(X,y,Sigma):
    """
    This function computes the mean and variance of the posterior
    """
    #TODO implement this
    mean = tikhonov_regression(X,y,Sigma)
    var=np.linalg.inv(X.T.dot(X)+np.linalg.inv(Sigma))
    mux = mean
    muy = mean
    sigmax = np.sqrt(var[0,0])
    sigmay = np.sqrt(var[-1,-1])
    sigmaxy = var[0,-1]
    return mux,muy,sigmax,sigmay,sigmaxy

Sigmas = [np.array([[1,0],[0,1]]), np.array([[1,0.25],[0.25,1]]),
          np.array([[1,0.9],[0.9,1]]), np.array([[1,-0.25],[-0.25,1]]),
          np.array([[1,-0.9],[-0.9,1]]), np.array([[0.1,0],[0,0.1]])]
names = [str(i) for i in range(1,6+1)]

for num_data in [5,50,500]:
    X,Y = generate_data(num_data)`
    for i,Sigma in enumerate(Sigmas):

        mux,muy,sigmax,sigmay,sigmaxy = compute_mean_var(X,Y,Sigma)
        # TODO compute the mean and covariance of posterior.

        x = np.arange(0.5, 1.5, 0.01)
        y = np.arange(0.5, 1.5, 0.01)
        X_grid, Y_grid = np.meshgrid(x, y)
        #X-grid

        Z = matplotlib.mlab.bivariate_normal(X_grid,Y_grid,mux,muy,sigmax,sigmay,sigmaxy)
        #,mux,muy,sigmaxy
        # TODO Generate the function values of bivariate normal.

        # plot
        plt.figure(figsize=(10,10))
        CS = plt.contour(X_grid, Y_grid, Z,
                         levels = np.concatenate([np.arange(0,0.05,0.01),np.arange(0.05,1,0.05)]))
        plt.clabel(CS, inline=1, fontsize=10)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Sigma'+ names[i] + ' with num_data = {}'.format(num_data))
        plt.savefig('Sigma'+ names[i] + '_num_data_{}.png'.format(num_data))
        #plt.show()
