### Project Description

This is an ongoing project, and the main task is to try different data-processing methods to improve the gene 
classification accuracy by deep learning method.


Definition of k-mer:

Contiguous k-letter substrings are called k-mers. For example, the string "ACGTACGGTA" has the following 3-mer: 
"ACG", "CGT","GTA", "TAC", "TAC", "ACG", "ACG", "CGG", "GGT", "GTA".

Definition of abword:

In a string, the substring of the form "abbbb..." is called an "abword". For example, the string "ACGTACGGTA" has the following abwords starting from "A":
"ACGT", "ACGGT"

Definition of Minimizer:

A (w,k)-minimizer is the minimal k-mer from w adjacent k-mers.
For example, we have a string "1234567", choose w = 5, and k=3. The we have 5 3-mers:
"123", "234", "345", "456", "567". From these 5 3-mer, we get one (5,3)-minimizer: "123".

see this paper for detail: [Reducing storage requirements for biological sequence comparison](https://academic.oup.com/bioinformatics/article/20/18/3363/202143)