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
import csv


def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        print('modeling to keras ',round((i/(len(X) - time_steps))*100,2), ('%'))
        v = X.iloc[i: (i+time_steps), 1:3].to_numpy()
        Xs.append(v)
        ys.append(y.iloc[i+time_steps])
    return np.array(Xs), np.array(ys)



def saveFile(dataset, name='dataset'):
    print('saving: ',name, '......')
    f = open(name,'w')
    try:
        writer = csv.writer(f)
        writer.writerow(dataset.columns)
        for i in np.arange(int(dataset.shape[0])):
            writer.writerow(dataset.iloc[i,])
    finally:
        f.close()


#carregando datasets
print('loading dataset')
test  = pd.read_csv('../sdn_test_normalized.csv', delimiter=",")
train  = pd.read_csv('../sdn_train_normalized.csv', delimiter=",")


print('creating window')
TIME_STEPS = 1

X_train,Y_train = create_dataset(train, train.delay, time_steps=TIME_STEPS)
X_test,Y_test = create_dataset(test, test.delay, time_steps=TIME_STEPS)

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


y_train_inv = scaler2.inverse_transform(Y_train)
y_test_inv = scaler2.inverse_transform(Y_test)


print('saving data')
cols = ['train_delay' ]

df_ds =  pd.DataFrame(y_train_inv, columns = cols)
saveFile(df_ds, name='sdn_delays_train_desnormalized.csv')

df_ds =  pd.DataFrame(y_test_inv, columns = cols)
saveFile(df_ds, name='sdn_delays_test_desnormalized.csv')



fig1 = plt.figure()
a1 = fig1.add_subplot(1,1,1)
a1.plot(Y_train, marker='.', label='true')
a1.legend();

fig2 = plt.figure()
a2 = fig2.add_subplot(1,1,1)
a2.plot(Y_test,'r',marker='.', label='predicted')
a2.legend();


fig3 = plt.figure()
a3 = fig3.add_subplot(1,1,1)
a3.plot(Y_test,'r',marker='.', label='predicted')
a3.legend();

