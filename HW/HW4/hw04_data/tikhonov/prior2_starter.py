import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
w = [1.0,1.0]
n_test = 100
n_trains = np.arange(5,205,5)
n_trails = 500

Sigmas = [np.array([[1,0],[0,1]]), np.array([[1,0.25],[0.25,1]]),
          np.array([[1,0.9],[0.9,1]]), np.array([[1,-0.25],[-0.25,1]]),
          np.array([[1,-0.9],[-0.9,1]]), np.array([[0.1,0],[0,0.1]])]
names = ['Sigma{}'.format(i+1) for i in range(6)]

#ef generate_data(n):
#    """
#    This function generates data of size n.
#    """
#    #TODO implement this
#    return (X,y)
#
#def tikhonov_regression(X,Y,Sigma):
#    """
#    This function computes w based on the formula of tikhonov_regression.
#    """
#    #TODO implement this
#    return w

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

def compute_mse(X,Y, w):
    """
    This function computes MSE given data and estimated w.
    """
    #TODO implement this
    mse = np.mean((np.squeeze(X,dot(w))-Y)**2)
    return mse

def compute_theoretical_mse(w):
    """
    This function computes theoretical MSE given estimated w.
    """
    #TODO implement this
    theoretical_mse = compute_mse(X_test,y_test,w)
    return theoretical_mse

# Generate Test Data.
X_test, y_test = generate_data(n_test)

mses = np.zeros((len(Sigmas), len(n_trains), n_trails))

theoretical_mses = np.zeros((len(Sigmas), len(n_trains), n_trails))

for seed in range(n_trails):
    np.random.seed(seed)
    for i,Sigma in enumerate(Sigmas):
        for j,n_train in enumerate(n_trains):
            #TODO implement the mses and theoretical_mses

# Plot
plt.figure()
for i,_ in enumerate(Sigmas):
    plt.plot(n_trains, np.mean(mses[i],axis = -1),label = names[i])
plt.xlabel('Number of data')
plt.ylabel('MSE on Test Data')
plt.legend()
plt.savefig('MSE.png')

plt.figure()
for i,_ in enumerate(Sigmas):
    plt.plot(n_trains, np.mean(theoretical_mses[i],axis = -1),label = names[i])
plt.xlabel('Number of data')
plt.ylabel('MSE on Test Data')
plt.legend()
plt.savefig('theoretical_MSE.png')


plt.figure()
for i,_ in enumerate(Sigmas):
    plt.loglog(n_trains, np.mean(theoretical_mses[i]-1,axis = -1),label = names[i])
plt.xlabel('Number of data')
plt.ylabel('MSE on Test Data')
plt.legend()
plt.savefig('log_theoretical_MSE.png')
