import numpy as np
import scipy.io
import copy

from scipy.optimize import leastsq
import matplotlib.pyplot as plt

mdict = scipy.io.loadmat("atrans.mat")

x = mdict['x']
u = mdict['u']
y = mdict['y']


# Your code to compute a and b
#def leastsq(x,y):
#    meanx = sum(x) / len(x)   
#    meany = sum(y) / len(y)   
#
#    xsum = 0.0
#    ysum = 0.0
#
#    for i in range(len(x)):
#        xsum += (x[i] - meanx)*(y[i]-meany)
#        ysum += (x[i] - meanx)**2
#
#    k = xsum/ysum
#    b = meany - k*meanx
#    return k,b
Xi=np.array(x[0])
Yi=np.array(y[0])
Ui=np.array(u[0])
 
 
def func(p,x):
    k,b=p
    return k*x+b
def error(p,x,y):
    return func(p,x)-y
 
p0=[1,20]
p1=[1,20]
Para=leastsq(error,p0,args=(Xi,Yi))

## This b is not true B, b=u[t]*TrueB 
k,b=Para[0]
print("k=",k,"b=",b)
print("y="+str(round(k,2))+"x+"+str(round(b,2)))
 
TB=np.array(b/Ui)
p0=[1,20]
 
Para1=leastsq(error,p1,args=(Xi,Ui))
sk,TruB=Para1[0]

print(TB)
plt.figure(figsize=(8,6)) 
plt.scatter(Xi,TB,color="blue",label="Dataset2",linewidth=2)

x1=np.linspace(11,17,100) 
y1=sk*x1+TruB 
plt.plot(x,y,color="red",label="Line",linewidth=2)
plt.legend() 
plt.show()

plt.figure(figsize=(8,6)) 
plt.scatter(Xi,Yi,color="green",label="Dataset",linewidth=2)
 
x=np.linspace(11,17,100) 
y=k*x+b 
plt.plot(x,y,color="red",label="Line",linewidth=2)
plt.legend() 
plt.show()


