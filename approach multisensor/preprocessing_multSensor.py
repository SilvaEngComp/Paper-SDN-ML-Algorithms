# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 11:46:21 2021

@author: silva
"""

import csv

import pandas as pd
import numpy as np
import time

def makingTimestamp(dataset):
    dataset.date = dataset.date+ ' ' +dataset.time
    #removing local from time
    for i in np.arange(dataset['date'].shape[0]):
        x = dataset.date.iloc[i].split('.')
        dataset.date.iloc[i] = x[0]
        print("\033[H\033[J") 
        print('making timestamp ',round((i/dataset['date'].shape[0])*100,2), ('%'))
    dataset.date= pd.to_datetime(dataset.date, infer_datetime_format=True)
    return dataset

def preprocessing(dataset):
    MyLIst = []
    dataset = makingTimestamp(dataset)
    for i in np.arange((dataset.shape[0])):
        print('preprocessing ',round((i/dataset.shape[0])*100, 2), ('%'))
        temperature = dataset.iloc[i, 4]
        label = 0
        dalay = 0
        timestamp = dataset.iloc[i, 0]

        MyLIst.append([timestamp,time.mktime(timestamp.timetuple()), temperature,label, dalay ])
    cols = ['timestamp', 'seconds', 'temperature','label' ,'delay' ]
    return pd.DataFrame(MyLIst, columns = cols)

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

#carregando dataset da simulação
print('loading dataset')
dataset = pd.read_csv('../data.txt', delimiter=" ")
subset  = dataset[(dataset["moteid"] >=2)]
subset  = subset[(subset["moteid"] <=6)]
subset = subset[(subset["temperature"] >=15)]
subset = subset[(subset["temperature"] <=26)]

#separa os datasets
print('separing dataset')
test  = subset[(subset["moteid"] <=3)]
train  = subset[subset["moteid"] > 3]


#preprocessando
print('init preprocessing')
train_ML = preprocessing(train)
test_ML = preprocessing(test)

#limitando delay
print('filtring delay')
#train_ML = train_ML[train_ML['delay'] < 20000 ]
#test_ML = test_ML[test_ML['delay'] < 20000 ]

#salvando datasets preprocessados
saveFile(train_ML, name='datasets/sem_concept_drift/sdn_train.csv')
saveFile(test_ML, name='datasets/sem_concept_drift/sdn_test.csv')


