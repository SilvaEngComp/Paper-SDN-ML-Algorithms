# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 16:41:18 2021

@author: silva
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras

import csv

import pandas as pd
import numpy as np
from timeit import default_timer as timer
from IPython.display import clear_output

from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
#Normalize functions

        

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []    
    start = timer()
    for i in range(len(X) - time_steps):
        clear_output(wait=True)
        print('modeling to keras ',round((i/(len(X) - time_steps))*100,2), ('%'), end='')
        s = round(timer() - start)
        if(s>60):
            s /=60
            print(' ', s, ' seconds')
        v = X.iloc[i: (i+time_steps), 0:2].to_numpy()
        Xs.append(v)
        ys.append(y.iloc[i+time_steps])
    return np.array(Xs), np.array(ys)

#train functions
def LSTMconf(X_train):
    print('Init config LSTM')
    model = keras.Sequential()
    model.add(
        keras.layers.Bidirectional(
            keras.layers.LSTM(
                 activation="relu",
                units=512,
                input_shape=(X_train.shape[1],X_train.shape[2])
            )
        ))
    model.add(keras.layers.Dense(units=512, activation="relu"))
    model.add(keras.layers.Dense(units=512, activation="relu"))
    model.add(keras.layers.Dense(units=512, activation="relu"))
    model.add(keras.layers.Dense(units=512, activation="relu"))
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.Dense(units=1))
    
    loss ="mse"
    optim = tf.keras.optimizers.Adam(
    learning_rate=0.0001)

    model.compile(loss=loss, optimizer=optim, 
             )
    return model

    
def LSTMfit():
    
    prep_dataset1 = pd.read_csv('../datasets/dataset_test_02_07.csv', delimiter=",")
    df = prep_dataset1.iloc[:,1:4]
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
    
    X_train,Y_train = create_dataset(train, train.delay)
    model = LSTMconf(X_train)
        
    print('Init Train')
    history = model.fit(
        X_train, Y_train, 
        epochs=256, 
        batch_size= 128,
        validation_split=0.1,
        shuffle=False,
    )

    
    print('Saving Model')
    model.save('lstm')
    
    predict(test, model)

def predict(test=None, model=None):
    
    if test is None:
         prep_dataset1 = pd.read_csv('../datasets/dataset_test_02_07.csv', delimiter=",")
         df = prep_dataset1.iloc[:,1:4]
         train_size = int(len(df) * 0.8)
         train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
         
    if model is None:
        model = keras.models.load_model('lstm')
   
    X_test,Y_test = create_dataset(test, test.delay)
    
    y_pred = model.predict(X_test)

    print(y_pred)
    print(Y_test)
   
    size = np.min([y_pred.shape[0],Y_test.shape[0] ])
    rmse =  mean_squared_error(Y_test[0:size], y_pred[0:size], squared=False)
    mae =  mean_absolute_error(Y_test[0:size], y_pred[0:size])
    median_mae = median_absolute_error(Y_test[0:size], y_pred[0:size])
    evs = explained_variance_score(Y_test[0:size], y_pred[0:size])

    print('MSE: ',rmse)
    print('MAE: ',mae)
    print('MEDMAE: ',median_mae)
    print('Explained Variance Score: ',evs)
    return np.mean(y_pred)

print(LSTMfit())