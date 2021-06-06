# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 23:47:45 2021

@author: silva
"""
import os
import csv

import pandas as pd
import numpy as np
from timeit import default_timer as timer

from sklearn.preprocessing import StandardScaler
from skmultiflow.drift_detection import PageHinkley

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


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
        print('creating concept window ',round((i/(len(X) - time_steps))*100,2), ('%'))
        v = X.iloc[i: (i+time_steps), 2].to_numpy()
        Xs.append(v)                     
    return np.array(Xs)

def getConceptDrifft(data_stream, dataset):
    ph = PageHinkley()

    # Adding stream elements to the PageHinkley drift detector and verifying if drift occurred
    CHANGE = 0
    lastconcept=0
    for i in np.arange(data_stream.shape[0]):  
        DELAY= (dataset.iloc[i, 1] - dataset.iloc[lastconcept, 1])
        for j in np.arange(data_stream[i].shape[0]):  
            ph.add_element(data_stream[i][j])
            if ph.detected_change():
                CHANGE = 1
                print('Change has been detected in data: ' + str(data_stream[i][j]) + ' - delay: '+str(DELAY) +' - of index: '+str(i)+'x'+str(i) )
                break


        init = i*WINDOW  

        if(CHANGE):   
            dataset.iloc[init: (init+WINDOW), 4] =  DELAY
            dataset.iloc[init: (init+WINDOW),3] =  CHANGE
            CHANGE = 0
            lastconcept = i

    return dataset
    

        
start = timer()
    
#carregando datasets
print('loading dataset')
test  = pd.read_csv('sdn_test.csv', delimiter=",")
train  = pd.read_csv('sdn_train.csv', delimiter=",")
WINDOW = 20
print(train)

#gerando concept drift train
print('loading train concept drift')
data_stream_train = create_window(train, time_steps=WINDOW)
print(data_stream_train)
train= getConceptDrifft(data_stream_train, train)

#gerando concept drift test
print('loading test concept drift')
data_stream_test = create_window(test, time_steps=WINDOW)
test= getConceptDrifft(data_stream_test, test)


#salvando datasets normalizados
saveFile(train, name='sdn_train_unnormalized.csv')
saveFile(test, name='sdn_test_unnormalized.csv')

#separando em treino e teste

print('loading Normalizing')
#colunas que serão normalizadas
f_columns = ['temperature','label']

#normalizando train
scaler1 = StandardScaler().fit(train[f_columns])
scaler2 = StandardScaler().fit(train[f_columns])

scaler1= scaler1.fit(train[f_columns].to_numpy())
scaler2 = scaler2.fit(train[['delay']])

train.loc[:,f_columns] = scaler1.transform(train[f_columns].to_numpy())
train['delay'] = scaler2.transform(train[['delay']])

#normalizando test
scaler3 = StandardScaler().fit(test[f_columns])
scaler4 = StandardScaler().fit(test[f_columns])
scaler3 = scaler3.fit(test[f_columns].to_numpy())
scaler4 = scaler4.fit(test[['delay']])

test.loc[:,f_columns] = scaler3.transform(test[f_columns].to_numpy())
test['delay'] = scaler4.transform(test[['delay']])


#salvando datasets normalizados
saveFile(train, name='sdn_train_normalized.csv')
saveFile(test, name='sdn_test_normalized.csv')

print('duração: ', timer() - start) 



