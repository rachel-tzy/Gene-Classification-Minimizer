import random
import csv
import numpy as np
import re
import pandas as pd

# alphbet = 'ACGT'
#
# three_mer_list = []
# for char1 in alphbet:
#     for char2 in alphbet:
#         for char3 in alphbet:
#             three_mer_list.append(char1 + char2 + char3)
#
# three_mer_hash_value = np.arange(64)
# random.shuffle(three_mer_hash_value)
#
# three_mer_dict = dict(zip(three_mer_list, three_mer_hash_value))
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


def find_w_mer(kmer, w):
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
        minimizer_list.append(find_w_mer(k_mer, w))
        i = i + 1
    return minimizer_list

s = 'ACGTAGATC'
print(find_minimizer(s, 5, 3))
