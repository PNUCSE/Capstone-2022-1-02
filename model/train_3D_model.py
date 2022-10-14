import os
import json
import glob
import random
import collections

import numpy as np
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import cv2 as cv
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow_hub as hub
from tensorflow import keras
from keras import layers
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical

import tensorflow.keras
from tensorflow.keras.applications import ResNet50

train_df = pd.read_csv("/home/bono/trsna/train_labels.csv")
test_df = pd.read_csv("/home/bono/trsna/test_labels.csv")


import os
for dirname, _, filenames in os.walk('/home/bono/jeon/64train'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
for dirname, _, filenames in os.walk('/home/bono/jeon/64test'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

tensorflow.compat.v1.disable_eager_execution()
mirrored_strategy = tensorflow.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])

def load_dicom(path):
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    data = data-np.min(data)
    if np.max(data) != 0:
        data = data/np.max(data)
    data = (data*255).astype(np.uint8)
    return data

def listdirs(folder):
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]

train_dir = '/home/bono/jeon/64train/'
trainset=[]
trainlabel=[]
trainidt=[]
for i in tqdm(range(len(train_df))):
    idt = train_df.loc[i,'BraTS21ID']
    path = os.path.join(train_dir)
    idt2 = ('00000'+str(idt))[-5:]

    filename = train_dir + idt2 + '-T1wCE.dcm'
    dcm = pydicom.dcmread(filename)
    frame_generator = pydicom.encaps.generate_pixel_data_frame(dcm.PixelData)

    if 'SamplesPerPixel' not in dcm:
      dcm.SamplesPerPixel = 1

    if 'PhotometricInterpretation' not in dcm:
      dcm.PhotometricInterpretation = 1

    if 'BitsStored' not in dcm:
      dcm.BitsStored = 1    
 
    for im in range(64):
      img=dcm.pixel_array[im]
      image=img_to_array(img)
      trainset+=[image]
      trainlabel+=[train_df.loc[i,'MGMT_value']]
      trainidt+=[idt]

test_dir='/home/bono/jeon/64test/'
testset=[]
testidt=[]
for i in tqdm(range(len(test_df))):
    idt = test_df.loc[i,'BraTS21ID']
    path = os.path.join(test_dir)
    idt2 = ('00000'+str(idt))[-5:]

    filename = test_dir + idt2 + '-T1wCE.dcm'
    dcm = pydicom.dcmread(filename)
    frame_generator = pydicom.encaps.generate_pixel_data_frame(dcm.PixelData)

    if 'SamplesPerPixel' not in dcm:
      dcm.SamplesPerPixel = 1

    if 'PhotometricInterpretation' not in dcm:
      dcm.PhotometricInterpretation = 1

    if 'BitsStored' not in dcm:
      dcm.BitsStored = 1

    for im in range(64):
      img=dcm.pixel_array[im]
      image=img_to_array(img)
      testset += [image]
      testidt += [idt]

y = np.array(trainlabel)
Y_train = to_categorical(y)
X_train = np.array(trainset)
X_test = np.array(testset)

base_model = ResNet50(include_top=False, weights=None, input_shape=(64, 64, 1))

x = base_model.output
x = tensorflow.keras.layers.GlobalAveragePooling2D()(x)
x = tensorflow.keras.layers.Dropout(0.7)(x)
predictions = tensorflow.keras.layers.Dense(2, activation= 'softmax')(x)
model = tensorflow.keras.models.Model(inputs=base_model.input, outputs=predictions)

model.summary()

opt = keras.optimizers.Adam(learning_rate=0.001)

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

hist = model.fit(X_train, Y_train, epochs=100, batch_size=1024, verbose=1)

get_ac = hist.history['accuracy']
get_los = hist.history['loss']

print("---------------------------------------")

epochs = range(len(get_ac))
plt.plot(epochs, get_ac, 'g', label='Accuracy of Training data')
plt.plot(epochs, get_los, 'r', label='Loss of Training data')
plt.title('Training data accuracy and loss')
plt.legend(loc=0)

plt.savefig('final_graph.png', facecolor='#eeeeee', edgecolor='black', format='png', dpi=200)

y_pred = model.predict(X_test)
pred = np.argmax(y_pred,axis=1)
result = pd.DataFrame(testidt)
result[1] = pred
result.columns = ['BraTS21ID','MGMT_value']
result2 = result.groupby('BraTS21ID', as_index=False).mean()
result2

print(result2)
result2.to_csv('final_submission1.csv', index=False)
print("---------------------------------------")

result2['MGMT_value'] = result2['MGMT_value'].round()
print(result2)
print("---------------------------------------")
result2.to_csv('final_submission2.csv', index=False)
