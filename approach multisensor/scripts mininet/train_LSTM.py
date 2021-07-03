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
from sklearn.preprocessing import StandardScaler
from IPython.display import clear_output

from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
#Normalize functions
def normalizing(df):
    f_columns = ['temperature']
    scaler1 = StandardScaler().fit(df)
    scaler2 = StandardScaler().fit(df)

    scaler1= scaler1.fit(df[f_columns].to_numpy())
    scaler2 = scaler2.fit(df[['delay']])

    df.loc[:,f_columns] = scaler1.transform(df[f_columns].to_numpy())
    df['delay'] = scaler2.transform(df[['delay']])
    return df


def unormalizing(df,Y_test,y_pred ):
    scaler = StandardScaler().fit(df)
    scaler = scaler.fit(df[['delay']])
    y_test_inv = scaler.inverse_transform(Y_test.reshape(1,-1))
    y_pred_inv = scaler.inverse_transform(y_pred)
    
    return y_test_inv, y_pred_inv

def saveFile(dataset, name='dataset'):
    print('saving: ',name, '......')
    f = open(name,'a')
    try:
        writer = csv.writer(f)
        writer.writerow(dataset.columns)
        for i in np.arange(int(dataset.shape[0])):
            writer.writerow(dataset.iloc[i,])
    finally:
        f.close()
        

def preprocessing(MyLIst): 
    cols = ['temperature', 'label' ,'delay' ]
    delta_df = pd.DataFrame(np.array(MyLIst), columns = cols)
    saveFile(delta_df, 'datasets/delta/sdn_train_mininet_unormalized.csv')
    norm = normalizing(delta_df)
    saveFile(norm, 'datasets/delta/sdn_train_mininet_normalized.csv')

    return norm

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []    
    for i in range(len(X) - time_steps):
        clear_output(wait=True)
        v = X.iloc[i: (i+time_steps), 0:1].to_numpy()
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

def checkFileExistance(filePath):
    try:
        with open(filePath, 'r') as f:
            return True
    except FileNotFoundError as e:
        return False
    except IOError as e:
        return False
    
def LSTMfit(window):
    
    train = preprocessing(window)
    X_train,Y_train = create_dataset(train, train.delay)  

    
    if(checkFileExistance('lstm.h5')):
        model = keras.models.load_model('models/lstm')
    else:
        model = LSTMconf(X_train)
        
    print('Init Train')
    history = model.fit(
        X_train, Y_train, 
        epochs=256, 
        batch_size= 128,
        validation_split=0.1,
        shuffle=False,
    )
    

    history = LSTMfit(model,X_train, Y_train)
    
    print('Saving Model')
    model.save('lstm.h5')
    return history

def predict(test):
    
    model = keras.models.load_model('../multisensor delta/models/lstm')
    test_norm = normalizing(test)
    X_test,Y_test = create_dataset(test_norm, test_norm.delay) 
    
    y_pred = model.predict(X_test)
    
    y_test_inv, y_pred_inv = unormalizing(test, Y_test, y_pred)
   
    size = np.min([y_pred_inv.shape[0],y_test_inv.shape[0] ])
    rmse =  mean_squared_error(y_test_inv[0:size], y_pred_inv[0:size], squared=False)
    mae =  mean_absolute_error(y_test_inv[0:size], y_pred_inv[0:size])
    median_mae = median_absolute_error(y_test_inv[0:size], y_pred_inv[0:size])
    evs = explained_variance_score(y_test_inv[0:size], y_pred_inv[0:size])

    print('MSE: ',rmse)
    print('MAE: ',mae)
    print('MEDMAE: ',median_mae)
    print('Explained Variance Score: ',evs)
    return y_pred

test = pd.read_csv('../datasets/dataset_test_02_07.csv', delimiter=",")
predict(test.iloc[0:50,1:4])
#r = []
 #   r.append(row)
 #   r = np.array(r)   
 #   y_pred = model.predict(r)