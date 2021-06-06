# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 03:13:19 2021

@author: silva
"""

import os
import csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []    
    for i in range(len(X) - time_steps):
        print('modeling to keras ',round((i/(len(X) - time_steps))*100,2), ('%'), end='')
        print(' ', round(timer() - start), ' seconds')
        v = X.iloc[i: (i+time_steps), 2:3].to_numpy()
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

    
start = timer()
#carregando datasets
print('loading dataset')
test  = pd.read_csv('../sdn_test_normalized.csv', delimiter=",")
train  = pd.read_csv('../sdn_train_normalized.csv', delimiter=",")

print('creating window')
TIME_STEPS = 1

X_train,Y_train = create_dataset(train, train.delay, time_steps=TIME_STEPS)
X_test,Y_test = create_dataset(test, test.delay, time_steps=TIME_STEPS)


#salvando datasets normalizados
saveFile(X_train, name='X_train.csv')
saveFile(Y_train, name='Y_train.csv')
saveFile(X_test, name='X_test.csv')
saveFile(Y_test, name='Y_test.csv')