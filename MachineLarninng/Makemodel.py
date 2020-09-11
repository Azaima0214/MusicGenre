# -*- coding: utf-8 -*-
"""Untitled24.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZWAt6aw2aoTup6OR2EoKPHjA2D0Dvft9
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from google.colab import drive
drive.mount('/content/drive')

import csv

root_dir = "/content/drive/My Drive/Colab Notebooks/"

csvfile = open(root_dir+"data2.csv")
title = csvfile.readlines()[0]
data = np.loadtxt(root_dir+"data2.csv", delimiter=',',skiprows=1)

import pandas as pd

X = data[:,:-1]
y = data[:, -1]
y = y.astype('int64')

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1,random_state= 0)

print(X_train.mean(axis=0))
print(X_train.std(axis=0))

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(512,activation='relu',input_shape=(X_train.shape[1],)))
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128,activation='relu'))
model.add(layers.Dense(128,activation='relu'))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(11,activation='softmax'))

model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train, 11)
y_test = np_utils.to_categorical(y_test, 11)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_test

history = model.fit(
    x=X_train,
    y=y_train,
    batch_size=256,
    epochs=3000,
    verbose=1,
    validation_data=(X_test,y_test)
)

model.save(root_dir+'model2.h5',include_optimizer=False)

from keras import models
model=models.load_model(root_dir+'model2.h5',compile=False)

a=[0.6050305697492526,-0.16892546253406088,0.06387765124744746,0.4309004050289814,0.10803706330009952,-0.1588579546623533,0.53424105729846,-0.07038221082374314,0.15145614915423147,-0.6057221101439892,1.2292176771521268,-0.6330175206694733,0.649908672864595,0.0840998984540684,-0.15603629295659616,-0.6104330378546773,0.9663021001839429,-0.6813917325308899,0.8123004434777638,0.5929258444903875,-0.6079421008901774,0.15263273281694148,0.0660346814485428,-0.37938151899135125,-0.9525500443162915,0.3044174344724825]
a=np.array(a).reshape(1,26)
label=['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock','alternative']
result = model.predict_on_batch(a)[0]
print(label[result.argmax()])