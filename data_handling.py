import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import random
import csv
import re

# path = '/tmp/tzy/Metagenomic-Data/label_dict.npy'
# path = 'C:\\Users\\TZY\\PycharmProjects\\Metagenomic Data\\label_dict.npy'
# path = 'C:\\Users\\TZY\\PycharmProjects\\Metagenomic Data\\Mgnify\\Collinsella\\label_dict.npy'
# path = '/tmp/tzy/Metagenomic-Data/Mgnify/Collinsella/label_dict.npy'
# LABEL_DICT = np.load(path, allow_pickle=True).item()

three_mer_dict = {'AAA': 62, 'AAC': 38, 'AAG': 50, 'AAT': 41, 'ACA': 52,
                  'ACC': 18, 'ACG': 45, 'ACT': 48, 'AGA': 26, 'AGC': 22,
                  'AGG': 27, 'AGT': 1, 'ATA': 59, 'ATC': 2, 'ATG': 51,
                  'ATT': 11, 'CAA': 17, 'CAC': 56, 'CAG': 5, 'CAT': 6,
                  'CCA': 60, 'CCC': 9, 'CCG': 21, 'CCT': 47, 'CGA': 8,
                  'CGC': 44, 'CGG': 19, 'CGT': 24, 'CTA': 33, 'CTC': 12,
                  'CTG': 53, 'CTT': 54, 'GAA': 39, 'GAC': 63, 'GAG': 30,
                  'GAT': 36, 'GCA': 49, 'GCC': 37, 'GCG': 34, 'GCT': 4,
                  'GGA': 14, 'GGC': 20, 'GGG': 42, 'GGT': 0, 'GTA': 10,
                  'GTC': 32, 'GTG': 7, 'GTT': 15, 'TAA': 35, 'TAC': 16,
                  'TAG': 43, 'TAT': 13, 'TCA': 28, 'TCC': 29, 'TCG': 40,
                  'TCT': 25, 'TGA': 46, 'TGC': 61, 'TGG': 57, 'TGT': 23,
                  'TTA': 3, 'TTC': 58, 'TTG': 55, 'TTT': 31}

def find_ab_word(s, a, n):
    # ab word is of the form abbbb..., the length of ab word is n+1
    # n >= 2
    alphabet = "ACGT"
    i = 0
    started = False
    word_list = []
    word = ''
    a_index_list = [substr.start() for substr in re.finditer(a, s)]
    for i in a_index_list:
        if a not in s[i+1:i+1+n]:
            word_list.append(s[i:i+1+n])
    return word_list

def find_w_mer_min(kmer, w):
    i = 0
    minimal = 100
    minimal_wmer = ''
    while i <= len(kmer) - w:
        if kmer[i:i+w] in three_mer_dict:
            w_mer_value = three_mer_dict[kmer[i:i+w]]
            if w_mer_value < minimal:
                minimal = w_mer_value
                minimal_wmer = kmer[i:i+w]
        i = i + 1
    return minimal_wmer

def find_minimizer(s, k, w):
    # minimizer is the mimimal w-mer in the k-mer
    # k >= w
    minimizer_list = []
    i = 0
    while i <= len(s) - k:
        k_mer = s[i:i+k]
        minimizer_list.append(find_w_mer_min(k_mer, w))
        i = i + 1
    return minimizer_list

def find_w_mer_max(kmer, w):
    i = 0
    maximum = -1
    maximum_wmer = ''
    while i <= len(kmer) - w:
        if kmer[i:i+w] in three_mer_dict:
            w_mer_value = three_mer_dict[kmer[i:i+w]]
            if w_mer_value > maximum:
                maximum = w_mer_value
                maximum_wmer = kmer[i:i+w]
        i = i + 1
    return maximum_wmer

def find_maxmizer(s, k, w):
    # minimizer is the mimimal w-mer in the k-mer
    # k >= w
    minimizer_list = []
    i = 0
    while i <= len(s) - k:
        k_mer = s[i:i+k]
        minimizer_list.append(find_w_mer_max(k_mer, w))
        i = i + 1
    return minimizer_list

def find_extremer(s, k, w):
    # extremers are the minimal nd max w-mer in the k-mer
    # k >= w
    extremer_list = []
    i = 0
    while i <= len(s) - k:
        k_mer = s[i:i+k]
        extremer_list.append(find_w_mer_min(k_mer, w))
        extremer_list.append(find_w_mer_max(k_mer, w))
        i = i + 1
    return extremer_list

# Reading data from training / testing files and returning one-hot-encodings to driver
def read_data_ab_word(LABEL_DICT, filename, num_word, num_classes, word_length):
    f = open(filename, 'r')
    # Create a list of the alphabet

    all_seq_one_hot = []
    labels = []
    alphabet_dict = {'C':1, 'G':2, 'T':3}
    for line in f:
        l = line.strip().split('\t')
        seq = l[0]
        category = l[1]
        word_list_A = find_ab_word(seq, 'A', word_length)

        # Convert characters to character indices
        index_seq = []
        for word in word_list_A:
            index = 1
            for char in word[1::]:
                if char not in alphabet_dict:
                    index = -1
                    break
                else:
                    index = alphabet_dict[char] * index
            index_seq.append(index-1)

        # Padding with index '-1' if the sequence is not long enough
        # NOTE: '-1' is also used as the index of unknown characters (padding can be treated as unknown anyway)
        if len(index_seq) < num_word:
            for _ in range(num_word - len(index_seq)):
                index_seq.append(-1)

        # Convert indices to one-hot-vectors, '-1' becomes the all-zeros vector
        index_seq_one_hot = []
        for i in index_seq:
            all_zeroes = [0] * (3**word_length)
            if i >= 0:
                all_zeroes[i] = 1
            index_seq_one_hot.append(all_zeroes)

        all_seq_one_hot.append(index_seq_one_hot)
        labels.append(category)

    x = np.array(all_seq_one_hot, dtype='float32')

    # Convert integer class labels to one-hot vectors
    all_labels_one_hot = []
    for i in labels:
        one_hot = [0] * num_classes
        one_hot[LABEL_DICT[i]] = 1
        all_labels_one_hot.append(one_hot)
    y = np.array(all_labels_one_hot, dtype='float32')

    return x, y

# Reading data from training / testing files and returning one-hot-encodings to driver
def read_data_minimizer(LABEL_DICT, filename, num_word, num_classes, k, w):
    f = open(filename, 'r')
    # Create a list of the alphabet

    all_seq_one_hot = []
    labels = []
    alphabet_dict = {'A':1, 'C':2, 'G':3, 'T':4}
    for line in f:
        l = line.strip().split('\t')
        seq = l[0]
        category = l[1]
        word_list = find_minimizer(seq, k, w)

        # Convert characters to character indices
        index_seq = []
        for word in word_list:
            index = 1
            for char in word[1::]:
                if char not in alphabet_dict:
                    index = -1
                    break
                else:
                    index = alphabet_dict[char] * index
            index_seq.append(index-1)
        # Padding with index '-1' if the sequence is not long enough
        # NOTE: '-1' is also used as the index of unknown characters (padding can be treated as unknown anyway)
        if len(index_seq) < num_word:
            for _ in range(num_word - len(index_seq)):
                index_seq.append(-1)

        # Convert indices to one-hot-vectors, '-1' becomes the all-zeros vector
        index_seq_one_hot = []
        for i in index_seq:
            all_zeroes = [0] * (4**w)
            if i >= 0:
                all_zeroes[i] = 1
            index_seq_one_hot.append(all_zeroes)

        all_seq_one_hot.append(index_seq_one_hot)
        labels.append(category)

    x = np.array(all_seq_one_hot, dtype='float32')

    # Convert integer class labels to one-hot vectors
    all_labels_one_hot = []
    for i in labels:
        one_hot = [0] * num_classes
        one_hot[LABEL_DICT[i]] = 1
        all_labels_one_hot.append(one_hot)
    y = np.array(all_labels_one_hot, dtype='float32')

    return x, y


# Reading data from training / testing files and returning one-hot-encodings to driver
def read_data_extremer(LABEL_DICT, filename, num_word, num_classes, k, w):
    f = open(filename, 'r')
    # Create a list of the alphabet

    all_seq_one_hot = []
    labels = []
    alphabet_dict = {'A':1, 'C':2, 'G':3, 'T':4}
    for line in f:
        l = line.strip().split('\t')
        seq = l[0]
        category = l[1]
        word_list = find_extremer(seq, k, w)

        # Convert characters to character indices
        index_seq = []
        for word in word_list:
            index = 1
            for char in word[1::]:
                if char not in alphabet_dict:
                    index = -1
                    break
                else:
                    index = alphabet_dict[char] * index
            index_seq.append(index-1)
        # Padding with index '-1' if the sequence is not long enough
        # NOTE: '-1' is also used as the index of unknown characters (padding can be treated as unknown anyway)
        if len(index_seq) < num_word:
            for _ in range(num_word - len(index_seq)):
                index_seq.append(-1)

        # Convert indices to one-hot-vectors, '-1' becomes the all-zeros vector
        index_seq_one_hot = []
        for i in index_seq:
            all_zeroes = [0] * (4**w)
            if i >= 0:
                all_zeroes[i] = 1
            index_seq_one_hot.append(all_zeroes)

        all_seq_one_hot.append(index_seq_one_hot)
        labels.append(category)

    x = np.array(all_seq_one_hot, dtype='float32')

    # Convert integer class labels to one-hot vectors
    all_labels_one_hot = []
    for i in labels:
        one_hot = [0] * num_classes
        one_hot[LABEL_DICT[i]] = 1
        all_labels_one_hot.append(one_hot)
    y = np.array(all_labels_one_hot, dtype='float32')

    return x, y

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


def generate_label_dict(directory):
    filePath = directory
    file_list = os.listdir(filePath)
    file_dict = {}
    label_count = 0
    for i in range(0, len(file_list)):
        if 'fna' in file_list[i]:
            fname = file_list[i].replace('.fna', '')
            file_dict[fname] = label_count
            label_count = label_count + 1
            print(directory + fname,
                  file=open(directory + '\\targets_addresses.txt', 'a'))
    print(file_dict)
    np.save(directory + '\\label_dict', file_dict)


def generate_seq_dict(directory):
    path = directory
    files = os.listdir(path)  # get names of all file in the directory
    seq = {}
    for file in files:
        if "fna" in file:
            print(file)
            f = open(path + '\\' + file, encoding='UTF-8')
            subseq = {}
            for line in f:
                if line.startswith('>'):
                    name = line.replace('>', '').split()[0]
                    subseq[name] = ''
                else:
                    subseq[name] += line.replace('\n', '').strip()
            seq[file] = subseq
    np.save(path + '\\10_species.npy', seq)


def generate_seq_dict():
    path = 'C:\\Users\\TZY\\PycharmProjects\\Metagenomic Data\\Mgnify\\Collinsella'
    files = os.listdir(path)  # get names of all file in the directory
    seq_dict_path = 'C:\\Users\\TZY\\PycharmProjects\\Metagenomic Data\\Mgnify\\Collinsella\\10_species.npy'
    seq_dict = np.load(seq_dict_path, allow_pickle=True).item()
    files_list = []
    file_genome = {}
    for file in files:
        if "fna" in file:
            files_list.append(file)
            file_genome[file] = []

    for file in files_list:
        f = open(path + '\\' + file, encoding='UTF-8')
        for line in f:
            if line.startswith('>'):
                file_genome[file].append(line)
    with open(path + "Collinsella-test.fna", "w") as file_output:
        num_seq = 1000
        for i in range(num_seq):
            rand_file = np.random.randint(len(files_list))
            rand_genome = np.random.randint(
                (len(file_genome[files_list[rand_file]])))
            rand_legth = np.random.randint(50, 200)
            # file_genome[files_list[rand_file]][rand_genome]
            name = file_genome[files_list[rand_file]][rand_genome].replace('>',
                                                                           '').split()[
                0]
            print(name[0:name.rindex('_')])
            seq = seq_dict[files_list[rand_file]][name]
            start = random.randint(0, len(seq) - rand_legth)
            subseq = seq[start: start + rand_legth]
            file_output.write(
                file_genome[files_list[rand_file]][rand_genome] + "\n")
            file_output.write(subseq + "\n")


if __name__ == '__main__':
    alphabet = "ACGT"
    seq_len = 200  # Fixed length of a sequence of chars
# num_classes = 50  # Num of categories/concepts
# file_name = 'C:\\Users\\TZY\\PycharmProjects\\Metagenomic Data\\10_species.npy'
# file_name = '/tmp/tzy/Metagenomic-Data/10_species.npy'
# file_name = 'C:\\Users\\TZY\\PycharmProjects\\Metagenomic Data\\Mgnify\\Collinsella\\10_species.npy'
# seq_dictionary = np.load(file_name, allow_pickle=True).item()
# generate_seq_tsv(seq_dictionary, 50, 200, 10**6)

# generate_label_dict('C:\\Users\\TZY\\PycharmProjects\\Metagenomic Data\\Mgnify\\10 genus')
# generate_seq_dict('C:\\Users\\TZY\\PycharmProjects\\Metagenomic Data\\Mgnify\\10 genus')
# path = 'C:\\Users\\TZY\\PycharmProjects\\Metagenomic Data\\Mgnify\\10 genus\\label_dict.npy'
# LABEL_DICT = np.load(path, allow_pickle=True).item()
# print(LABEL_DICT)
