# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 11:46:21 2021

@author: silva
"""

import csv

import pandas as pd
import numpy as np
import time
from IPython.display import clear_output

def makingTimestamp(dataset):
    dataset.date = dataset.date+ ' ' +dataset.time
    #removing local from time
    for i in np.arange(dataset['date'].shape[0]):
        x = dataset.date.iloc[i].split('.')
        dataset.date.iloc[i] = x[0]
        clear_output(wait=True)
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
dataset = pd.read_csv('../../../data.txt', delimiter=" ")

print('separing dataset')
train1  = dataset[(dataset["moteid"] ==2)]
train2  = dataset[(dataset["moteid"] ==3)]
train3  = dataset[(dataset["moteid"] ==4)]

test1  = dataset[(dataset["moteid"] ==5)]



train1 = train1[(train1["temperature"] >=15)]
train1 = train1[(train1["temperature"] <=26)]

train2 = train2[(train2["temperature"] >=15)]
train2 = train2[(train2["temperature"] <=26)]

train3 = train3[(train3["temperature"] >=15)]
train3 = train3[(train3["temperature"] <=26)]

test1 = test1[(test1["temperature"] >=15)]
test1 = test1[(test1["temperature"] <=26)]


#preprocessando
print('init preprocessing')
train_ML1 = preprocessing(train1)
train_ML2 = preprocessing(train2)
train_ML3 = preprocessing(train3)

test_ML1 = preprocessing(test1)

#limitando delay
print('filtring delay')
#train_ML = train_ML[train_ML['delay'] < 20000 ]
#test_ML = test_ML[test_ML['delay'] < 20000 ]

#salvando datasets preprocessados
saveFile(train_ML1, name='../datasets/sem_concept_drift/sdn_train1.csv')
saveFile(train_ML2, name='../datasets/sem_concept_drift/sdn_train2.csv')
saveFile(train_ML3, name='../datasets/sem_concept_drift/sdn_train3.csv')
saveFile(test_ML1, name='../datasets/sem_concept_drift/sdn_test1.csv')


