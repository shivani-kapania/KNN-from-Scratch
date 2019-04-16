#!/usr/bin/env python
# coding: utf-8




import numpy as np
from matplotlib import pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')




def distance(x1, x2):
    d = np.sqrt(((x1-x2)**2).sum())
    return d

def knn(X_train, y_train, xt, k=7):
    vals = []
    for ix in range(X_train.shape[0]):
        d = distance(X_train[ix], xt)
        vals.append([d, y_train[ix]])
    sorted_labels = sorted(vals, key=lambda z: z[0])
    neighbours = np.asarray(sorted_labels)[:k, -1]
    
    freq = np.unique(neighbours, return_counts=True)
    
    return freq[0][freq[1].argmax()]



import pandas as pd
import datetime




df = pd.read_csv('train.csv')
df.head()




data = df.values[:2000]
print(data.shape)



split = int(0.8 * data.shape[0])

X_train = data[:split, 1:]
X_test = data[split:, 1:]

y_train = data[:split, 0]
y_test = data[split:, 0]

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)



plt.figure(0)
plt.imshow(X_train[90].reshape((28, 28)), cmap='gray', interpolation='none')
print(y_train[90])
plt.show()



def get_acc(kx):
    preds = []
    # print kx
    for ix in range(X_test.shape[0]):
        start = datetime.datetime.now()
        preds.append(knn(X_train, y_train, X_test[ix], k=kx))
        # print(datetime.datetime.now() - start)
    preds = np.asarray(preds)
    
    # print(preds.shape)
    return 100*float((y_test == preds).sum())/preds.shape[0]




for ix in range(2, 20):
    print("k:", ix, "| Acc:", get_acc(ix))


# In[ ]:




