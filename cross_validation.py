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

train_df = pd.read_csv("/home/bono/trsna/train_labels.csv")
sample_df = pd.read_csv("/home/bono/trsna/test_labels.csv")

import tensorflow_hub as hub
from pydicom.pixel_data_handlers.util import apply_voi_lut
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical

import tensorflow.keras
from tensorflow.keras.applications import ResNet50
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn import metrics

# seed
global_seed = 42

# training fold
total_fold = 5
fold = 0

tensorflow.compat.v1.disable_eager_execution()
mirrored_strategy = tensorflow.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])


def load_dicom(path):
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data


class estimator:
    _estimator_type = ''
    classes_ = []

    def __init__(self, model, classes):
        self.model = model
        self._estimator_type = 'classifier'
        self.classes_ = classes

    def predict(self, X):
        y_prob = self.model.predict(X)
        # y_pred = y_prob.argmax(axis=1)
        print(y_prob)
        y_pred = np.argmax(y_prob, axis=1)
        print(y_pred)
        print(np.shape(y_pred))
        # y_pred = np.argmax(y_pred, axis=0)
        # print(np.shape(y_pred))
        # print(y_pred)
        return y_pred


def listdirs(folder):
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]


# 5 fold
for fold in range(5):
    # id 0001 label 0
    train_set = pd.read_csv(f'/home/bono/k-fold-cv/train_splits-5/split-{fold}/train_{fold}.csv')
    test_set = pd.read_csv(f'/home/bono/k-fold-cv/validation_splits-5/split-{fold}/test_{fold}.csv')
    train_set_subj_id = list(train_set['participant_id'])
    train_set_labels = list(train_set['diagnosis'])
    test_set_subj_id = list(test_set['participant_id'])
    test_set_labels = list(test_set['diagnosis'])
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    for i in train_set_subj_id:
        orig_i = i
        i = ('00000' + str(i))[-5:]
        total_path = os.path.join('/home/bono/resized_224/train', i, 'T1wCE')  # data_folder_pata
        sub_file = os.listdir(total_path)
        for sub in sub_file:
            img = cv.imread(os.path.join(total_path, sub))
            img = cv.resize(img, (64, 64))
            image = img_to_array(img)
            # img = np.expand_dims(img, axis=0)
            Y_train += list(train_set[train_set['participant_id'] == orig_i]['diagnosis'])
            X_train += [img]
    for j in test_set_subj_id:
        # same
        orig_j = j
        j = ('00000' + str(j))[-5:]
        total_path = os.path.join('/home/bono/resized_224/train', j, 'T1wCE')
        sub_file = os.listdir(total_path)
        for sub in sub_file:
            img = cv.imread(os.path.join(total_path, sub))
            img = cv.resize(img, (64, 64))
            image = img_to_array(img)
            # img = np.expand_dims(img, axis=0)
            X_test += [img]
            Y_test += list(test_set[test_set['participant_id'] == orig_j]['diagnosis'])

    X_train = np.array(X_train)
    Y_train = to_categorical(np.array(Y_train))
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    
    base_model = ResNet50(include_top=False, weights=None, input_shape=(64, 64, 3))

    x = base_model.output
    x = tensorflow.keras.layers.GlobalAveragePooling2D()(x)
    x = tensorflow.keras.layers.Dropout(0.7)(x)
    predictions = tensorflow.keras.layers.Dense(2, activation='softmax')(x)
    model = tensorflow.keras.models.Model(inputs=base_model.input, outputs=predictions)


    opt = keras.optimizers.Adam(learning_rate=0.001)

    model.compile(loss="categorical_crossentropy",
                  optimizer=opt, metrics=["accuracy"])

    callback = tensorflow.keras.callbacks.EarlyStopping(monitor='loss', patience=8)

    hist = model.fit(X_train, Y_train, epochs=100, batch_size=1024, verbose=1, callbacks=[callback])

    classifier = estimator(model, classes=['pos', 'neg'])

    plot_confusion_matrix(classifier, X_test, Y_test, cmap=plt.cm.BuPu)
    plt.savefig(f'/home/bono/CM/Confusion_Matrix_fold{fold}.png')


