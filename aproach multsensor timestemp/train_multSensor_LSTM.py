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
from skmultiflow.drift_detection import PageHinkley

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        print('modeling to keras ',round((i/(len(X) - time_steps))*100,2), ('%'))
        v = X.iloc[i: (i+time_steps), 0:2].to_numpy()
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
 
def create_window(X, time_steps=5):
    Xs = []
    for i in range(len(X) - time_steps):
        print('creating concept window ',(round(i/(len(X) - time_steps))*100,2), ('%'))
        v = X.iloc[i: (i+time_steps), 0].to_numpy()
        Xs.append(v)                     
    return np.array(Xs)

def getConceptDrifft(data_stream, dataset):
    ph = PageHinkley()

    # Adding stream elements to the PageHinkley drift detector and verifying if drift occurred
    CHANGE = 0
    lastconcept=0
    for i in np.arange(data_stream.shape[0]):  
        DELAY= (dataset.iloc[i, 0] - dataset.iloc[lastconcept, 0]).total_seconds()
        for j in np.arange(data_stream[i].shape[0]):  
            ph.add_element(data_stream[i][j])
            if ph.detected_change():
                CHANGE = 1
                print('Change has been detected in data: ' + str(data_stream[i][j]) + ' - delay: '+str(DELAY) +' - of index: '+str(i)+'x'+str(i) )
                break


        init = i*WINDOW  

        if(CHANGE):   
            dataset.iloc[init: (init+WINDOW), 2] =  DELAY
            dataset.iloc[init: (init+WINDOW),1] =  CHANGE
            CHANGE = 0
            lastconcept = i

    return dataset
    

#carregando datasets
print('loading dataset')

print('separing dataset')
test  = pd.read_csv('sdn_test.csv', delimiter=" ")
train  = pd.read_csv('sdn_train.csv', delimiter=" ")
WINDOW = 20

#gerando concept drift train
print('loading train concept drift')
data_stream_train = create_window(train, time_steps=WINDOW)
train= getConceptDrifft(data_stream_train, train)

#gerando concept drift test
print('loading test concept drift')
data_stream_test = create_window(test, time_steps=WINDOW)
test= getConceptDrifft(data_stream_test, test)


#separando em treino e teste

print('loading Normalizing')
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
print('Init Train')
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
print('Saving Model')
model.save('models/m2')

