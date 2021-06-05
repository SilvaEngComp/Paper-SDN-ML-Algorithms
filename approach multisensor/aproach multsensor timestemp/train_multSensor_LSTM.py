# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 23:47:45 2021

@author: silva
"""
import os
import tensorflow as tf
from tensorflow import keras

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i: (i+time_steps), 1:3].to_numpy()
        Xs.append(v)
        ys.append(y.iloc[i+time_steps])
    return np.array(Xs), np.array(ys)
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        print('modeling to keras ',round((i/(len(X) - time_steps))*100,2), ('%'))
        v = X.iloc[i: (i+time_steps), 0:2].to_numpy()
        Xs.append(v)
        ys.append(y.iloc[i+time_steps])
    return np.array(Xs), np.array(ys)


    

#carregando datasets
print('loading dataset')
test  = pd.read_csv('../sdn_test_normalized.csv', delimiter=",")
train  = pd.read_csv('../sdn_train_normalized.csv', delimiter=",")

print('creating window')
TIME_STEPS = 5

X_train,Y_train = create_dataset(train, train.delay, time_steps=TIME_STEPS)
X_test,Y_test = create_dataset(test, test.delay, time_steps=TIME_STEPS)

print(X_train.shape)
#configurando rede para treinamento
print('Init Train')
model = keras.Sequential()
model.add(
    keras.layers.Bidirectional(
        keras.layers.LSTM(
            units=128,
            input_shape=(X_train.shape[1],X_train.shape[2])
        )
    ))

#model.add(keras.layers.Dense(units=40))
#model.add(keras.layers.Dense(units=40))
#model.add(keras.layers.Dense(units=40))
#model.add(keras.layers.Dense(units=40))
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.Dense(units=1))

loss ="mse"
optim = tf.keras.optimizers.Adam(
    learning_rate=0.0001)
metrics=["accuracy"]

model.compile(loss=loss, optimizer=optim, 
             metrics=metrics
             )

#treinando 

history = model.fit(
    X_train, Y_train, 
    epochs=10, 
    batch_size= 64,
    validation_split=0.25,
    shuffle=False,
#     callbacks=[tensorboard_callback]
)

#salvando modelo
print('Saving Model')
model.save('models/m2')


fig1 = plt.figure()
ax1 = fig1.add_subplot(1,1,1)
ax1.plot(history.history['loss'], label='train')
ax1.plot(history.history['val_loss'], label='validation')
ax1.legend();



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


y_pred = model.predict(X_test)

y_test_inv = scaler2.inverse_transform(Y_test)
y_pred_inv = scaler2.inverse_transform(y_pred)

print('teste')
print(y_test_inv)

print('\n\n pred')
print(y_pred_inv)
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
a4.plot(y_pred_inv.flatten(),'r',marker='.', label='predicted')
a4.plot(y_test_inv.flatten(), marker='.', label='true')
a4.legend();
