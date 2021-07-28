import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.models import load_model
import random
import csv
import tensorflow as tf
from data_handling import read_data_ab_word,generate_seq_tsv

gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

num_dataset = 100
history_path = './data/Mgnify-5-genus-prediction-50to200_history_abword.npy'
# history_path = 'C:\\Users\\TZY\\PycharmProjects\\Metagenomic Data\\Mgnify\\' \
#                   '5genus\\train\\Mgnify-5-genus-prediction-50to200_history_abword.npy'
history = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}
history = np.load(history_path, allow_pickle=True).item()
acc_train = history['accuracy']
acc_val = history['val_accuracy']
loss_train = history['loss']
loss_val = history['val_loss']

train_file_path = './data/Mgnify-5-genus-prediction-50to200_train.tsv'
label_path = '/tmp/tzy/Metagenomic-Data/Mgnify/5genus/train/5genus_label_dict.npy'
seq_path = '/tmp/tzy/Metagenomic-Data/Mgnify/5genus/train/5genus_dict.npy'
# train_file_path = 'C:\\Users\\TZY\\PycharmProjects\\Metagenomic Data\\Mgnify\\' \
#                   '5genus\\train\\Mgnify-5-genus-prediction-50to200_train.csv'
# label_path = 'C:\\Users\\TZY\\PycharmProjects\\Metagenomic Data\\Mgnify\\' \
#                   '5genus\\train\\label_dict.npy'
# seq_path = 'C:\\Users\\TZY\\PycharmProjects\\Metagenomic Data\\Mgnify\\' \
#                   '5genus\\train\\5genus_dict.npy'
model_name = '5genus-prediction-model.hdf5'
LABEL_DICT = np.load(label_path, allow_pickle=True).item()
num_word = 40
num_classes = 5
word_length = 4
num_seq = 10000
for i in range(num_dataset):
    print('Training time: ', i)
    generate_seq_tsv(train_file_path, seq_path, 50, 200, 50000)

    data = read_data_ab_word(LABEL_DICT, train_file_path, num_word, num_classes, word_length)

    X, Y = data[0], data[1]
    # #Split training set and test set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)

    #Define the model
    model = load_model(model_name, compile=False)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    epochs = 3
    batch_size = 1000

    history = model.fit(X, Y, epochs=epochs, batch_size=batch_size,validation_split=0.1,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
    acc_train.extend(history.history['accuracy'])
    acc_val.extend(history.history['val_accuracy'])
    loss_train.extend(history.history['loss'])
    loss_val.extend(history.history['val_loss'])
    model.save(model_name)
    accr = model.evaluate(X_test,Y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
    history = {'loss': loss_train, 'val_loss': loss_val, 'accuracy': acc_train,
               'val_accuracy': acc_val}
    np.save(history_path, history)
