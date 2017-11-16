# Chao Fan
# hw4 coding part
# PS: Please change the path when you run the program. Thanks
# coding: utf-8

# In[1]:

import numpy as np

train = np.loadtxt('/Users/FAN/Desktop/CSE_COURSE/CSE151/data/hw4atrain.txt')
train_data = train[:, :-1]
train_label = train[:, -1]

test = np.loadtxt('/Users/FAN/Desktop/CSE_COURSE/CSE151/data/hw4atest.txt')
test_data = test[:, :-1]
test_label = test[:, -1]

train_label[train_label == 0] = -1
train_label[train_label == 6] = 1
test_label[test_label == 0] = -1
test_label[test_label == 6] = 1


# In[2]:

#perceptron algorithm
def perceptron(x, y, times):
    dim = x.shape[1]
    w = np.zeros((1, dim))
    for t1 in range(0, times):
        for t in range(0, x.shape[0]):
            if y[t] * np.dot(w, x[t]) <= 0:
                w = w + y[t] * x[t]
        t1 = t1 + 1
    return w


# In[3]:

# using reversive to predict value of test data x
def predict_perceptron(x, w):
    if x.ndim > 1:
        return np.array([predict_perceptron(sub_x, w) for sub_x in x]).reshape((-1,))
    test_value = np.dot(w, x)
    return np.sign(test_value)


# In[4]:

def calculate_error(y, y_predict):
    dim_y = y.shape[0]
    num_error = np.sum((y != y_predict).astype(float))
    return num_error/dim_y


# In[5]:

print "Perceptron result as follows:"
for i in range(1, 4):
    w = perceptron(train_data, train_label, i)
    
    y_predict = predict_perceptron(train_data, w)
    train_error = calculate_error(train_label, y_predict)
    
    y_predict = predict_perceptron(test_data, w)
    test_error = calculate_error(test_label, y_predict)
    
    print str(i) + " pass: train_error=" + str(train_error) + " test_error=" + str(test_error)


# In[6]:

#voted perceptron algorithm
def perceptron_voted(x, y, times):
    dim_x = x.shape[1]
    w = np.zeros((1, dim_x))
    c = np.ones((1,1))
    for _ in range(0, times):
        for t in range(0, x.shape[0]):
            if y[t] * np.dot(w[-1, :], x[t]) <= 0:
                w_new = w[-1, :] + y[t] * x[t]
                c_new = np.ones((1,1))

                #push w_new into w, push c_new into c, 
                w = np.vstack((w, w_new))
                c = np.vstack((c, c_new))
            else:
                c[-1] = c[-1] + 1
    return w, c


# In[7]:

def predict_perceptron_voted(x, w, c):
    if x.ndim > 1:
        return np.array([predict_perceptron_voted(sub_x, w, c) for sub_x in x]).reshape((-1,))
    sign_value = np.sign(np.dot(w,x))
    average_value = np.sign(np.dot(sign_value, c))
    return average_value


# In[8]:

print "Voted Perceptron result as follows:"
for i in range(1, 4):
    w, c = perceptron_voted(train_data, train_label, i)
    
    y_predict = predict_perceptron_voted(train_data, w, c)
    train_error = calculate_error(train_label, y_predict)
    
    y_predict = predict_perceptron_voted(test_data, w, c)
    test_error = calculate_error(test_label, y_predict)
    print str(i) + " pass: train_error=" + str(train_error) + " test_error=" + str(test_error)


# In[9]:

#average perceptron algorithm
def predict_perceptron_averaged(x, w, c):
    if x.ndim > 1:
        return np.array([predict_perceptron_averaged(sub_x, w, c) for sub_x in x]).reshape((-1,))
    inner_product = np.dot(c.T,w)
    predict = np.sign(np.dot(inner_product, x))
    return predict


# In[10]:

print "Average Perceptron result as follows:"
for i in range(1, 4):

    w, c = perceptron_voted(train_data, train_label, i)
    
    y_predict = predict_perceptron_averaged(train_data, w, c)
    train_error = calculate_error(train_label, y_predict)
    
    y_predict = predict_perceptron_averaged(test_data, w, c)
    test_error = calculate_error(test_label, y_predict)
    print str(i) + " pass: train_error=" + str(train_error) + " test_error=" + str(test_error)


# In[11]:

train_b = np.loadtxt('/Users/FAN/Desktop/CSE_COURSE/CSE151/data/hw4btrain.txt')
train_b_data = train_b[:, :-1]
train_b_label = train_b[:, -1]

test_b = np.loadtxt('/Users/FAN/Desktop/CSE_COURSE/CSE151/data/hw4btest.txt')
test_b_data = test_b[:, :-1]
test_b_label = test_b[:, -1]