import random
import csv
import numpy as np
import re
import pandas as pd
import string

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
alphbet = {'A':1,'C':2, 'G':3,'T':4}
def find_w_mer_min(kmer, w):
    i = 0
    minimal = 1000000000
    minimal_wmer = ''
    while i <= len(kmer) - w:
        if kmer[i:i+w] in three_mer_dict:
            w_mer_value = alphbet[kmer[i:i+w][0]]*100 + alphbet[kmer[i:i+w][1]]*10 + alphbet[kmer[i:i+w][2]]
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

k = 5
w = 3
num_trail = 100
num_kmer = 10000
result_kmer = []
result_count = []
result_hash = []

# for trail in range(num_trail):
#     fixed_kmer = ''.join([random.choice(['A','C','G','T']) for i in range(5)])
#     fixed_kmer_minimizer = find_w_mer_min(fixed_kmer, w)
#
#     # print(''.join([random.choice(['A','C','G','T']) for i in range(5)]))
#     count = 0
#     for i in range(num_kmer):
#         kmer = ''.join([random.choice(['A','C','G','T']) for i in range(5)])
#         kmer_minimizer = find_w_mer_min(kmer, w)
#         if kmer_minimizer == fixed_kmer_minimizer:
#             count = count + 1
#     result_kmer.append(fixed_kmer_minimizer)
#     result_count.append(count)

for wmer in three_mer_dict:
    count = 0
    for i in range(num_kmer):
        kmer = ''.join([random.choice(['A','C','G','T']) for i in range(5)])
        kmer_minimizer = find_w_mer_min(kmer, w)
        if kmer_minimizer == wmer:
            count = count + 1
    result_kmer.append(wmer)
    result_count.append(count)
    result_hash.append(alphbet[wmer[0]]*100 + alphbet[wmer[1]]*10 + alphbet[wmer[2]])
dataframe = pd.DataFrame({'hash': result_hash, 'minimizer':result_kmer,'count':result_count})

#将DataFrame存储为csv,index表示是否显示行名，default=True
dataframe.to_csv("prob_minimizer.csv",index=False,sep=',')

