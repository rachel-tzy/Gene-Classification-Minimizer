import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import random
import csv
import tensorflow as tf
from data_handling import read_data_ab_word, generate_seq_tsv

gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


test_file_path = './data/Mgnify-5-genus-prediction-50to200_test.tsv'
label_path = '/tmp/tzy/Metagenomic-Data/Mgnify/5genus/test/5genus_label_dict.npy'
seq_path = '/tmp/tzy/Metagenomic-Data/Mgnify/5genus/test/5genus_dict.npy'
model_name = '5genus-prediction-model.hdf5'
number_sample = 1000

# test_file_path = 'C:\\Users\\TZY\PycharmProjects\\LSTM-classification\\data\\Mgnify-5-genus-prediction-50to200_test.tsv'
# label_path = 'C:\\Users\\TZY\\PycharmProjects\\Metagenomic Data\\Mgnify\\5genus\\test\\5genus_label_dict.npy'
# seq_path = 'C:\\Users\\TZY\\PycharmProjects\\Metagenomic Data\\Mgnify\\5genus\\test\\5genus_dict.npy'

model_name = '5genus-prediction-model.hdf5'
LABEL_DICT = np.load(label_path, allow_pickle=True).item()
num_word = 40
num_classes = 5
word_length = 4


generate_seq_tsv(test_file_path, seq_path, 50, 200, number_sample)
data = read_data_ab_word(LABEL_DICT, test_file_path, num_word, num_classes, word_length)
X, Y = data[0], data[1]

model = load_model(model_name, compile=False)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
predictions = model.predict(X)
predictions_label = np.argmax(predictions, axis=1)
# print(np.argmax(predictions, axis=1))
count_2 = 0
predictions_two_labels = np.argsort(predictions, axis=1)[:, -2::]
for i in range(predictions_two_labels.shape[0]):
    if predictions_label[i] in predictions_two_labels[i, :]:
        count_2 = count_2 + 1

print("Total number of predicted samples: ", number_sample)
print("Number of corrected predicted samples: ", np.sum(predictions_label == np.argmax(Y, axis=1)))
print("Correction rate: ", np.sum(predictions_label == np.argmax(Y, axis=1))/number_sample)
print("Correction rate of two labels: ", count_2/number_sample)
