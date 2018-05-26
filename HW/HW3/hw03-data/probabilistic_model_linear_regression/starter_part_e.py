import numpy as np
import matplotlib.pyplot as plt

sample_size = [5,25,125,625]
plt.figure(figsize=[12, 10])             
for k in range(4):
    n = sample_size[k]
    X = np.random.normal(0,0.1,n)

    Z = np.random.uniform(-0.5,0.5,n)
    w = 10
    Y = w*X+Z

    B = np.linspace(0,4,num=50)
    W = np.linspace(0,4,num=50)
    N = [50,50]

    # generate data
    # np.linspace, np.random.normal and np.random.uniform might be useful functions

    
    likelihood = np.ones(N) # likelihood as a function of w

    for i1 in range(N[0]):
        w = W[i1]
        for i2 in range(N[1]):
            if(abs(Y[i2]-w*X[i2]>0.5)):
                likelihood[i1][i2] = 0
        # compute likelihood

    likelihood /= sum(likelihood) # normalize the likelihood
    
    plt.figure()
    # plotting likelihood for different n
    plt.plot(W, likelihood)
    plt.xlabel('w', fontsize=10)
    plt.title(['n=' + str(n)], fontsize=14)

plt.show()