# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 00:31:08 2021

@author: silva
"""


from tensorflow import keras
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i: (i+time_steps), 0:2].to_numpy()
        Xs.append(v)
        ys.append(y.iloc[i+time_steps])
    return np.array(Xs), np.array(ys)

model = keras.models.load_model('models/m2')
df = pd.read_csv('sdn_test.csv')


#normalizando
f_columns = ['temperature','label']
scaler1 = StandardScaler().fit(df)
scaler2 = StandardScaler().fit(df)

scaler1= scaler1.fit(df[f_columns].to_numpy())
scaler2 = scaler2.fit(df[['delay']])

df.loc[:,f_columns] = scaler1.transform(df[f_columns].to_numpy())
df['delay'] = scaler2.transform(df[['delay']])

#modelando para predizer
TIME_STEPS = 1
X_test,Y_test = create_dataset(df, df.delay, time_steps=TIME_STEPS)


#predizendo para o dataset de test
y_pred = model.predict(X_test)

#desfazendo normalização
y_test_inv = scaler2.inverse_transform(Y_test.reshape(1,-1))
y_pred_inv = scaler2.inverse_transform(y_pred)

#comparando resultado predito com real
plt.plot(y_test_inv.flatten(), marker='.', label='true')
plt.plot(y_pred_inv.flatten(),'r',marker='.', label='predicted')

plt.legend();


#avaliando erro de predição
print(mean_squared_error(y_test_inv[0], y_pred_inv[:,0]))

