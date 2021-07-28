import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import random
import csv
import re
import itertools

def find_ab_word(s, a, n):
    # ab word is of the form abbbb..., the length of ab word is n
    # n >= 2
    alphabet = "ACGT"
    i = 0
    word_list = []
    a_index_list = [substr.start() for substr in re.finditer(a, s)]
    for i in a_index_list:
        if a not in s[i+1:i+n] and len(s[i+1:i+n]) == n-1:
            word_list.append(s[i:i+n])
    return word_list

def check_legal_word(word):
    alphabet = 'ACGT'
    for char in word:
        if char not in alphabet:
            return False
    return True

def generate_all_abwords(word_length):
    alphabet = 'ACGT'
    index = 0
    word_to_index = {}
    for char in alphabet:
        sub_alphabet = alphabet.replace(char, '')
        for i in itertools.product(sub_alphabet, repeat=word_length-1):
            word_to_index[char + ''.join(i)] = index
            index = index + 1
    return word_to_index

# Reading data from training / testing files and returning one-hot-encodings to driver
def read_data_abword_frequency(LABEL_DICT, filename, word_length):
    f = open(filename, 'r')
    # Create a list of the alphabet
    alphabet_dict = {'A': {'C': 1, 'G': 2, 'T': 3},
                     'C': {'A': 1, 'G': 2, 'T': 3},
                     'G': {'C': 1, 'A': 2, 'T': 3},
                     'T': {'C': 1, 'G': 2, 'A': 3}}


    labels = []
    word_to_index = generate_all_abwords(word_length)
    frequency = [0]*len(word_to_index)
    all_seq_freq = []
    for line in f:
        frequency = [0] * len(word_to_index)
        l = line.strip().split('\t')
        seq = l[0]
        category = l[1]
        word_list_dict = {'A': find_ab_word(seq, 'A', word_length),
                          'C': find_ab_word(seq, 'C', word_length),
                          'G': find_ab_word(seq, 'G', word_length),
                          'T': find_ab_word(seq, 'T', word_length)}
        # Convert characters to character indices
        index_seq = []
        for start in word_list_dict:
            for word in word_list_dict[start]:
                if check_legal_word(word):
                    frequency[word_to_index[word]] = frequency[word_to_index[word]] + 1
        all_seq_freq.append(frequency)
        labels.append(LABEL_DICT[category])
    x = np.array(all_seq_freq)
    y = labels
    return all_seq_freq, y

def generate_seq_tsv(output_path, seq_path, min_len, max_len, num_seq):
    seq_dict = np.load(seq_path, allow_pickle=True).item()
    name_list = list(seq_dict.keys())
    print(name_list)
    file_name = output_path
    i = 0
    with open(file_name, 'w', newline='') as f:
        tsv_w = csv.writer(f, delimiter='\t')
        while i < num_seq:
            name = random.choice(name_list)
            subname = random.choice(list(seq_dict[name].keys()))
            lenth = random.randint(min_len, max_len)
            start = random.randint(0, len(seq_dict[name][subname]) - min_len)
            seq = seq_dict[name][subname][start:start + lenth]
            name = name.replace('.fna', '')
            tsv_w.writerow([seq, name])
            i = i + 1



if __name__ == '__main__':
    alphabet = "ACGT"
    seq_len = 200  # Fixed length of a sequence of chars
    num_classes = 10  # Num of categories/concepts
# file_name = 'C:\\Users\\TZY\\PycharmProjects\\Metagenomic Data\\10_species.npy'
# file_name = '/tmp/tzy/Metagenomic-Data/10_species.npy'
    file_name = 'C:\\Users\\TZY\\PycharmProjects\\Metagenomic Data\\Mgnify\\10genus\\train\\10genus_dict.npy'
    seq_dictionary = np.load(file_name, allow_pickle=True).item()
    out_put = 'C:\\Users\\TZY\\PycharmProjects\\Metagenomic Data\\Mgnify\\10genus\\train\\Mgnify-10genus-prediction-50to200_train.tsv'
    # generate_seq_tsv(out_put, file_name, 50, 200, 10)
    path = 'C:\\Users\\TZY\\PycharmProjects\\Metagenomic Data\\Mgnify\\10genus\\train\\10genus_label_dict.npy'
    LABEL_DICT = np.load(path, allow_pickle=True).item()
    x, y = read_data_abword_frequency(LABEL_DICT, out_put, 4)
    print(x)
    print(y)
# generate_label_dict('C:\\Users\\TZY\\PycharmProjects\\Metagenomic Data\\Mgnify\\10 genus')
# generate_seq_dict('C:\\Users\\TZY\\PycharmProjects\\Metagenomic Data\\Mgnify\\10 genus')
# path = 'C:\\Users\\TZY\\PycharmProjects\\Metagenomic Data\\Mgnify\\10 genus\\label_dict.npy'
# LABEL_DICT = np.load(path, allow_pickle=True).item()
# print(LABEL_DICT)
