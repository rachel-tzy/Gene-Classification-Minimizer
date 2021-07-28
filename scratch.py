import random
import csv
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from data_handling import read_data_ab_word,generate_seq_tsv


# test_file_path = 'C:\\Users\\TZY\PycharmProjects\\LSTM-classification\\data\\Mgnify-5-genus-prediction-50to200_test.csv'
# label_path = 'C:\\Users\\TZY\\PycharmProjects\\Metagenomic Data\\Mgnify\\5genus\\test\\5genus_label_dict.npy'
# seq_path = 'C:\\Users\\TZY\\PycharmProjects\\Metagenomic Data\\Mgnify\\5genus\\test\\5genus_dict.npy'
#
# model_name = '5genus-prediction-model.hdf5'
# LABEL_DICT = np.load(label_path, allow_pickle=True).item()
# num_word = 40
# num_classes = 5
# word_length = 4
#
#
# generate_seq_tsv(test_file_path, seq_path, 50, 200, 10)
#
# data = read_data_ab_word(LABEL_DICT, test_file_path, num_word, num_classes, word_length)
#
# print(np.argmax(data[1], axis=1))


# history_path = 'C:\\Users\\TZY\\PycharmProjects\\Minimizer\\data\\' \
#                'Mgnify-5-genus-prediction-50to200_history_abword.npy'
# history = np.load(history_path, allow_pickle=True).item()
# print(history['val_accuracy'])
#
# acc_train = history['accuracy']
# acc_val = history['val_accuracy']
# loss_train = history['loss']
# loss_val = history['val_loss']
# plt.title('Loss')
# plt.plot(loss_train, label='train')
# plt.plot(loss_val, label='test')
# plt.legend()
# plt.show()
#
# plt.title('Accuracy')
# plt.plot(acc_train, label='train')
# plt.plot(acc_val, label='test')
# plt.legend()
# plt.show()

def generate_seq_dict(directory):
    path = 'C:\\Users\\tzy\\PycharmProjects\\Metagenomic Data\\large-scale-metagenomics-1.0\\data' \
       '\\train-dataset\\'
    f = open(directory, encoding='UTF-8')
    subseq = {}
    for line in f:
        if line.startswith('>'):
            name = line.replace('>', '').split()[0]
            subseq[name] = ''
        else:
            subseq[name] += line.replace('\n', '').strip()
    np.save(path + '\\train_small_db.npy', subseq)
path = 'C:\\Users\\tzy\\PycharmProjects\\Metagenomic Data\\large-scale-metagenomics-1.0\\data' \
       '\\train-dataset\\train_small-db.fasta'
generate_seq_dict(path)
