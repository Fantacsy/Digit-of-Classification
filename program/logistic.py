
# coding: utf-8

# In[20]:

import csv
from math import *

train3 = [line.strip().split() for line in open("/Users/chaofan/dropbox/csecourse/cse250a/data/newTrain3.txt", 'r')]
for l in range (0, len(train3)):
    train3[l] = [float(i) for i in train3[l]]

train5 = [line.strip().split() for line in open("/Users/chaofan/dropbox/csecourse/cse250a/data/newTrain5.txt", 'r')]
for l in range (0, len(train5)):
    train5[l] = [float(i) for i in train5[l]]

    
test3 = [line.strip().split() for line in open("/Users/chaofan/dropbox/csecourse/cse250a/data/newTest3.txt", 'r')]
for l in range (0, len(test3)):
    test3[l] = [float(i) for i in test3[l]]
    
test5 = [line.strip().split() for line in open("/Users/chaofan/dropbox/csecourse/cse250a/data/newTest5.txt", 'r')]
for l in range (0, len(test5)):
    test5[l] = [float(i) for i in test5[l]]


# In[21]:

def sigmod(z):
    return 1.0 / (1.0 + exp(-z))

def dot(x,y):
    product = 0
    for i in range(len(x)):
        product = x[i] * y[i]
    return product

def Gradient(x0, x1, w):
    step = 1e-4
    g = [0] * len(w)
    
    for i in range(len(x0)):
        d = 0 - sigmod(dot(x0[i], w))
        for j in range(len(w)):
            g[j] += d * x0[i][j]
            
    for i in range(len(x1)):
        d = 1 - sigmod(dot(x1[i], w))
        for j in range(len(w)):
            g[j] += d * x1[i][j]
            
    for i in range(len(w)):
        w[i] += step * g[i]
    return w

def Likelihood(x0, x1, w):
    l = 0
    for i in range(len(x0)):
        l += log(sigmod(-dot(x0[i],w)))
    for i in range(len(x1)):
        l += log(sigmod(dot(x1[i], w)))
    return l


# In[22]:

w3 = [0]*64
w5 = [0]*64
l3 = []
l5 = []

for i in range(2500):
    w3 = Gradient(train5, train3, w3)
    w5 = Gradient(train3, train5, w5)
    likeilhood3 = Likelihood(train5, train3, w3)
    likeilhood5 = Likelihood(train5, train3, w3)    
    l3.append(likeilhood3)
    l5.append(likeilhood5)
print w3
print w5


# In[28]:

def ErrorRate(x0, x1, w0, w1):
    count = 0
    for i in range(len(x0)):
        if sigmod(dot(x0[i], w0)) > sigmod(dot(x0[i], w1)):
            count += 1
    for i in range(len(x1)):
        if sigmod(dot(x1[i], w1)) > sigmod(dot(x1[i], w0)):
            count += 1
    return 1.0 * count / (len(x0)+len(x1))


# In[29]:

print ErrorRate(train3, train5, w3, w5)
print ErrorRate(test3, test5, w3, w5)


# In[15]:

print l3

# In[16]:

print l5


# In[32]:

for i in range(0,8):
    print w3[i*8 + 0], w3[i*8 + 1],w3[i*8 + 2],w3[i*8 + 3],w3[i*8 + 4],w3[i*8 + 5],w3[i*8 + 6],w3[i*8 + 7] 


# In[33]:

for i in range(0,8):
    print w5[i*8 + 0], w5[i*8 + 1],w5[i*8 + 2],w5[i*8 + 3],w5[i*8 + 4],w5[i*8 + 5],w5[i*8 + 6],w5[i*8 + 7] 



