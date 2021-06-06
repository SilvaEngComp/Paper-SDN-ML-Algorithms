# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 12:49:17 2021

@author: silva
"""
from tensorflow import keras

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from timeit import default_timer as timer

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i: (i+time_steps), 0:2].to_numpy()
        Xs.append(v)
        ys.append(y.iloc[i+time_steps])
    return np.array(Xs), np.array(ys)

    
start = timer()
#carregando datasets
print('loading dataset')
X_train  = pd.read_csv('X_train.csv', delimiter=",")
Y_train  = pd.read_csv('Y_train.csv', delimiter=",")
X_test  = pd.read_csv('X_test.csv', delimiter=",")

Y_test  = pd.read_csv('Y_test.csv', delimiter=",")

test  = pd.read_csv('../sdn_test_normalized.csv', delimiter=",")
train  = pd.read_csv('../sdn_train_normalized.csv', delimiter=",")


#desnormalizing
#normalizando train
f_columns = ['temperature','label']
scaler1 = StandardScaler().fit(train[f_columns])
scaler2 = StandardScaler().fit(train[f_columns])

scaler1= scaler1.fit(train[f_columns].to_numpy())
scaler2 = scaler2.fit(train[['delay']])


#normalizando test
scaler3 = StandardScaler().fit(test[f_columns])
scaler4 = StandardScaler().fit(test[f_columns])

scaler3 = scaler3.fit(test[f_columns].to_numpy())
scaler4 = scaler4.fit(test[['delay']])


model = keras.models.load_model('models/mpl2')
y_pred = model.predict(X_test)

y_test_inv = scaler4.inverse_transform(Y_test)
y_pred_inv = scaler4.inverse_transform(y_pred)


print('mse: ', mean_squared_error(y_test_inv, y_pred_inv))




fig2 = plt.figure()
a2 = fig2.add_subplot(1,1,1)
a2.plot(y_test_inv.flatten(), marker='.', label='true')
a2.legend();

fig3 = plt.figure()
a3 = fig3.add_subplot(1,1,1)
a3.plot(y_pred_inv.flatten(),'r',marker='.', label='predicted')
a3.legend();

fig4 = plt.figure()
a4 = fig4.add_subplot(1,1,1)
a4.plot(y_test_inv.flatten(), marker='.', label='true')
a4.plot(y_pred_inv.flatten(),'r',marker='.', label='predicted')

a4.legend();
