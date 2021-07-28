import numpy as np
from data_handling_abword import read_data_abword_frequency, generate_seq_tsv
from sklearn import svm
import joblib
import os
test_file_path = 'C:\\Users\\TZY\\PycharmProjects\\Metagenomic Data\\Mgnify\\' \
                  '10genus\\test\\Mgnify-10genus-prediction-50to200_test.csv'

test_seq_path = 'C:\\Users\\TZY\\PycharmProjects\\Metagenomic Data\\Mgnify\\' \
                  '10genus\\test\\10genus_dict.npy'

label_path = 'C:\\Users\\TZY\\PycharmProjects\\Metagenomic Data\\Mgnify\\10genus\\test\\' \
             '10genus_label_dict.npy'

if not os.path.exists(label_path):
    label_path = '/tmp/tzy/Metagenomic-Data/Mgnify/10genus_73/test/10genus_label_dict.npy'
if not os.path.exists(test_file_path):
    test_file_path = '/tmp/tzy/Metagenomic-Data/Mgnify/10genus_73/test/' \
                 'Mgnify-10genus-prediction-50to200_test.csv'
if not os.path.exists(test_seq_path):
    test_seq_path = '/tmp/tzy/Metagenomic-Data/Mgnify/10genus_73/test/10genus_dict.npy'

LABEL_DICT = np.load(label_path, allow_pickle=True).item()
word_length = 4
num_seq = 1000
clf = joblib.load("abword_freq_svm_10genus_73.m")
generate_seq_tsv(test_file_path, test_seq_path, 50, 200, num_seq)
x_test, y_test = read_data_abword_frequency(LABEL_DICT, test_file_path, word_length)

# print(clf.decision_function(x_test[0]))
# print(clf.predict(x_test))
# print(y_test)
predicted = np.array(clf.predict(x_test))
real = np.array(y_test)
print('Accuracy: ', np.sum(predicted == real)/num_seq)
