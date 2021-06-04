# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 23:47:45 2021

@author: silva
"""
import os
import tensorflow as tf
from tensorflow import keras
import csv

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def preprocessing(dataset):
    MyLIst = []

    for i in np.arange((dataset.shape[0])):
        temperature = dataset.iloc[i, 1]
        label = dataset.iloc[i, 2]
        delay = dataset.iloc[i, 3]

        MyLIst.append([ temperature,label, delay ])
    cols = [ 'temperature','label' ,'delay' ]
    return pd.DataFrame(MyLIst, columns = cols)

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i: (i+time_steps), 0:2].to_numpy()
        Xs.append(v)
        ys.append(y.iloc[i+time_steps])
    return np.array(Xs), np.array(ys)

def saveFile(dataset, name='dataset'):
    f = open(name,'w')
    try:
        writer = csv.writer(f)
        writer.writerow(dataset.columns)
        for i in np.arange(int(dataset.shape[0])):
            writer.writerow(dataset.iloc[i,])
    finally:
        f.close()


#carregando dataset da simulação
dataset = pd.read_csv('dataset_moteid-01.csv')

#preprocessando
df = preprocessing(dataset)
saveFile(df, name='sdn_ml.csv')

#separando em treino e teste
train_size = int(len(df) * 0.9)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]

#colunas que serão normalizadas
f_columns = ['temperature','label']

#normalizando train
scaler1 = StandardScaler().fit(train)
scaler2 = StandardScaler().fit(train)

scaler1= scaler1.fit(train[f_columns].to_numpy())
scaler2 = scaler2.fit(train[['delay']])

train.loc[:,f_columns] = scaler1.transform(train[f_columns].to_numpy())
train['delay'] = scaler2.transform(train[['delay']])

#normalizando test
scaler3 = StandardScaler().fit(test)
scaler4 = StandardScaler().fit(test)
scaler3 = scaler3.fit(test[f_columns].to_numpy())
scaler4 = scaler4.fit(test[['delay']])

test.loc[:,f_columns] = scaler3.transform(test[f_columns].to_numpy())
test['delay'] = scaler4.transform(test[['delay']])

#colocando no formado do Keras pra treinamento
TIME_STEPS = 1

X_train,Y_train = create_dataset(train, train.delay, time_steps=TIME_STEPS)
X_test,Y_test = create_dataset(test, test.delay, time_steps=TIME_STEPS)

#configurando rede para treinamento

model = keras.Sequential()
model.add(
    keras.layers.Bidirectional(
        keras.layers.LSTM(
            units=40,
            input_shape=(X_train.shape[1],X_train.shape[2])
        )
    ))

model.add(keras.layers.Dense(units=40))
model.add(keras.layers.Dense(units=40))
model.add(keras.layers.Dense(units=40))
model.add(keras.layers.Dense(units=40))
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
model.save('models/m1')

