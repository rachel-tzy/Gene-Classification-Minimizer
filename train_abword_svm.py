import numpy as np
from data_handling_abword import read_data_abword_frequency, generate_seq_tsv
from sklearn import svm
import joblib
import os

train_file_path = 'C:\\Users\\TZY\\PycharmProjects\\Metagenomic Data\\Mgnify\\' \
                  '10genus\\train\\Mgnify-10genus-prediction-50to200_train.csv'

train_seq_path = 'C:\\Users\\TZY\\PycharmProjects\\Metagenomic Data\\Mgnify\\' \
                  '10genus\\train\\10genus_dict.npy'

label_path = 'C:\\Users\\TZY\\PycharmProjects\\Metagenomic Data\\Mgnify\\10genus\\train\\' \
             '10genus_label_dict.npy'
if not os.path.exists(train_file_path):
    train_file_path = '/tmp/tzy/Metagenomic-Data/Mgnify/10genus_73/train/Mgnify-10genus-prediction-50to200_train.csv'

if not os.path.exists(train_seq_path):
    train_seq_path = '/tmp/tzy/Metagenomic-Data/Mgnify/10genus_73/train/10genus_dict.npy'

if not os.path.exists(label_path):
    label_path = '/tmp/tzy/Metagenomic-Data/Mgnify/10genus_73/train/10genus_label_dict.npy'
num_seq = 50000
generate_seq_tsv(train_file_path, train_seq_path, 50, 200, num_seq)
LABEL_DICT = np.load(label_path, allow_pickle=True).item()
word_length = 4

x, y = read_data_abword_frequency(LABEL_DICT, train_file_path, 4)

if os.path.exists("abword_freq_svm_10genus_73.m"):
    clf = joblib.load("abword_freq_svm_10genus_73.m")
else:
    clf = svm.SVC(kernel='linear', C=10, gamma=0.5, decision_function_shape='ovo')
clf.fit(x, y)
joblib.dump(clf, "abword_freq_svm_10genus_73.m")
