from __future__ import division
# CIS 511 NLP - Assignment 1 - Collocation Identification

"""
Created on Sat Feb 2 20:30:18 2019

@author: Siyu Yang
@unique name: siyuya
@UMID:76998080
"""

import string
import math
import sys

#=== Functions ===#

#   1)  Load Files and create unigram dictionary
def create_unigram(filename):
    unigram_dict = {}
    unigram_num = 0
    with open(filename,'r') as f:
        for line in f:
            splitLine = line.split()
            for word in splitLine:               
                # store unigram (no tokens of only punctuation)
                if word not in string.punctuation:
                    #count the number of unigrams
                    unigram_num += 1
                    if word in unigram_dict:
                        unigram_dict[word] += 1
                    else:
                        unigram_dict[word] = 1
    return unigram_dict,unigram_num
    
#   2)  Load Files and create bigram dictionary    
def create_bigram(filename):
    bigram_dict = {}
    bigram_num = 0
    pre_word = "."
    with open(filename,'r') as f:
        for line in f:
            splitLine = line.split()
            for word in splitLine:
                if word not in string.punctuation:
                    # store bigram (no tokens of only punctuation)
                    if pre_word not in string.punctuation:
                        #count the number of bigrams
                        bigram_num += 1
                        bigram = pre_word + ' ' + word
                        if bigram in bigram_dict:
                            bigram_dict[bigram] += 1
                        else:
                            bigram_dict[bigram] = 1
                            
                # set next pre_word
                pre_word = word
                
    # discard bigrams that occur less than 5 times
    bigram_dict = { bigram:count for bigram, count in bigram_dict.items() if count >=5 }
    return bigram_dict, bigram_num



#   3)  Create Bigram Matrices
def create_matrices(bigram_dict,N):
    '''
    @param:
    bigram_dict: the dictionary of bigram
    N: the total number of the bigrams
    
    @return:
    matrices: the dictionary of the matrix for every bigram
    word1_dict: the dictionary of the first word
    word2_dict: the dictionary of the second word
    '''
    # create dictionaries and save word1 and word2 counts in dictionaries
    word1_dict = {}
    word2_dict = {}

    # loop through every bigram in the dictionary
    for bigram,value in bigram_dict.items():
        word1 = bigram.split()[0]
        word2 = bigram.split()[1]      
        # store word1
        if word1 in word1_dict:
            word1_dict[word1] += 1
        else:
            word1_dict[word1] = 1       
        # store word2
        if word2 in word2_dict:
            word2_dict[word2] += 1
        else:
            word2_dict[word2] = 1


    # create dictionary and store matrix for each bigram
    matrices = {}

    for bigram,value in bigram_dict.items():        
        # extract each word1 & word2
        word1 = bigram.split()[0]
        word2 = bigram.split()[1]
        # 1st value: both word1 and word2 (bigram)
        a = bigram_dict[bigram] 
        # 2rd value: only word2 occurences
        b = word2_dict[word2] - a 
        # 3rd value: only word1 occurences
        c = word1_dict[word1] -a 
        # 4th value: non word1 or word2
        d = N-a-b-c 
        
        #store values as matrix      
        matrix=[a,c,b,d]
        matrices[bigram] = matrix
       
    return matrices, word1_dict, word2_dict


#   4)  Calculate Chi-Square Score
def calculate_chi_square(matrices,N):
    '''
    @param:
    matrices: the dictionary of the matrix for every bigram
    N: the total number of the bigrams
    
    @return:
    chi_square_list: the dictionary of the chi-square scores
    '''
    chi_square_list = {}

    for bigram, matrix in matrices.items():        
        a = matrix[0]
        b = matrix[1]
        c = matrix[2]
        d = matrix[3]
        #calculate the chi-square score 
        chi_square = (N * pow((a*d-b*c),2))/((a+b)*(a+c)*(b+d)*(c+d))
        # store score in dictionary with bigram as key
        chi_square_list[bigram] = chi_square

    return chi_square_list


#   5)  Calculate PMI Score
def calculate_PMI(matrices, N, N1, bigram_dict, word1_dict, word2_dict):
    '''
    @param:
    matrices: the dictionary of the matrix for every bigram
    bigram_dict: the dictionary of bigram
    N: the total number of the bigrams
    N1: the total number of the unigrams
    word1_dict: the dictionary of the first word
    word2_dict: the dictionary of the second word
    
    @return:
    chi_square_list: the dictionary of the PMI scores
    '''
    PMI_list = {}

    for bigram,matrix in matrices.items():
        word1 = bigram.split()[0]
        word2 = bigram.split()[1]
        
        # calculate PMI score
        p = bigram_dict[bigram]/N
        p1 = word1_dict[word1]/N1
        p2 = word2_dict[word2]/N1
        PMI = math.log(p/(p1*p2))
        # store score in dictionary with bigram as key
        PMI_list[bigram] = PMI
        
    return PMI_list
  

if __name__ == "__main__":

    # 1)  Load files, create dictionaries for bigrams and unigrams
    bigram_dict, N = create_bigram(sys.argv[1])
    unigram_dict, N1 = create_unigram(sys.argv[1])
    
    # 2)  Load measure type
    measure= sys.argv[2]

    # 3) create bigram matrices and dictionaries for word1 and word2
    matrices, word1_dict, word2_dict = create_matrices(bigram_dict,N)
    
    # 4) calculate requested measurement score
    if measure == "chi-square":
        score = calculate_chi_square(matrices,N)
    if measure == "PMI":
        score = calculate_PMI(matrices, N, N1, bigram_dict, word1_dict, word2_dict)

    # 5) output top 20 ranked bigrams and scores
    top20 = sorted(score.items(), key=lambda item:item[1], reverse = True)[:20]
    for key,value in top20:
        print(key,value)

