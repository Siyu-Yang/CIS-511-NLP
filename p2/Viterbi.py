
# CIS 511 NLP - Assignment 2 - Viterbi Part-of-speech Tagger

"""
Created on Sat Feb 25 08:06:51 2020

@author: Siyu Yang
@unique name: siyuya
@UMID:76998080
"""

import numpy as np
import sys

#=== Functions ===#

#   1)  Load Train File
def load_train_data(trainFile):
    
    # collect and store raw counts required for algorithm

    tagFreqs_dict = {} # store corresponding frequency for each unique tag (KEY: tag, VALUE: count)
    wordFreqs_dict = {} # store corresponding frequency for each unique word (KEY: word, VALUE: count)
    tag_index = 0
    word_index = 0
    
    train_sample = []
    
    with open(trainFile) as data:
        for line in data:
            line = "<s>/<s>" + ' ' + line # process files as sentences
            splitline = line.strip().split() # add beginning
            
            new_line = []
            
            for pair in splitline:
                pair = pair.split('/')
                if len(pair) == 2:
                    new_line.append(pair)
                    
            train_sample.append(new_line)
            
            for i in range(len(new_line)):
                pre_tag = new_line[i][1]
                
                if pre_tag not in tagFreqs_dict: # add unique tag into dict
                    tagFreqs_dict[pre_tag] = tag_index
                    tag_index += 1                
                
                word = new_line[i][0]

                if word not in wordFreqs_dict: # add unique word into dict
                    wordFreqs_dict[word] = word_index
                    word_index += 1
 
    
    wordFreqs_dict['NON'] = word_index
    return tagFreqs_dict, wordFreqs_dict, train_sample


#   1)  Process Train File

def process_train_data(train_sample,tagFreqs_dict, wordFreqs_dict):

    tag_num = len(tagFreqs_dict) # total number of tags
    word_num = len(wordFreqs_dict) # total number of words
    
    tag_to_tag = np.ones((tag_num, tag_num))
    tag_to_word = np.ones((tag_num, word_num))
    
    
    
    for new_line in train_sample:
        for i in range(len(new_line)):
            word = new_line[i][0]
            pre_tag = new_line[i][1]
            tag_to_word[tagFreqs_dict[pre_tag],wordFreqs_dict[word]] += 1
            if i+1 < len(new_line):
                next_tag = new_line[i+1][1]
                tag_to_tag[tagFreqs_dict[pre_tag], tagFreqs_dict[next_tag]] += 1 
    
    tag_to_word = (tag_to_word/tag_to_word.sum(axis=1, keepdims=1)) 
    tag_to_tag = tag_to_tag/tag_to_tag.sum(axis=1, keepdims=1) 

    return tag_to_word, tag_to_tag


#   3)  Load Test File

def load_test_data(testFile):
    test_sample = []
    with open(testFile) as data:
        for line in data:
            line = "<s>/<s>" + ' ' + line 
            splitline = line.strip().split()
            new_line = []
            for pair in splitline:
                pair = pair.split('/')
                if len(pair) == 2:
                    new_line.append(pair)
            test_sample.append(new_line)
    return test_sample
    
    
    
#   4)  Viterbi Algorithm  
  
def Viterbi(test_sample,tag_to_word,tag_to_tag, tagFreqs_dict, wordFreqs_dict):
    # create storage for sentence tag predictions and true
    sen_samples_predit = []
    sen_samples_true = []

    # iterate through every test sentence
    for instance in test_sample:
    
        # 0) create storage for this sentence's predictions, probability scores, and back tag pointer
        
        prob_score = np.zeros((len(tagFreqs_dict),len(instance)))
        back_tag = np.zeros((len(tagFreqs_dict),len(instance)))
        label_list = np.zeros(len(instance))
        
        # 1) Initialization
        tag_init = '<s>'
        word = instance[1][0]
        
        if word not in wordFreqs_dict:
            word = 'NON'
        
        label_list[1] = (tagFreqs_dict[instance[1][1]])
        
        word_given_tag_prob = tag_to_word[:, wordFreqs_dict[word]]
        tag_given_tag_prob = tag_to_tag[0,:]
        
        prob_score[:,1] = list(word_given_tag_prob * tag_given_tag_prob)  # list 
        back_tag[:,1] = np.zeros((len(tagFreqs_dict))) # index 
        
        # 2) Iteration
        for i in range(2,len(instance)):
            word = instance[i][0] 
            
            max_score = []  # max score for each one 
            max_back_tag = []   # max score postion
             
            if word not in wordFreqs_dict:
                word = 'NON'
            label_list[i] = (tagFreqs_dict[instance[i][1]]) 
            
            transition_score = np.array(prob_score[:, i-1]).reshape(-1, 1) * tag_to_tag 
            
            for j in range(len(tagFreqs_dict)):  # for each tag 
                
                word_pre_prob = tag_to_word[j, wordFreqs_dict[word]]
                
                max_score.append(max(transition_score[:, j] * word_pre_prob))   # score for this tag 
                max_back_tag.append(np.argmax(transition_score[:, j] * word_pre_prob))
            
            prob_score[:, i] = np.array(max_score)
            back_tag[:, i] = np.array(max_back_tag)
            
        # 3) store predictions for this sentence    
        
        final_max = np.argmax(prob_score[:,-1])
        
        predict_label = np.zeros((len(instance)))
        
        predict_label[-1] = int(final_max)
        
        for i in range(len(instance)-2,0,-1):
            final_max = int(back_tag[final_max,i+1])
            predict_label[i] = int(final_max)


        sen_samples_predit.append(predict_label)
        sen_samples_true.append(label_list)
    
    return sen_samples_predit, sen_samples_true


#   5)  Calculate Predict Score  

def predict(sen_samples_predit,sen_samples_true ):
    # create storage for sentence tag predictions
    
    sentence_num = 0
    tag_num = 0
    correct_sentence_num = 0 
    wrong_tag_num = 0
    for sentence in range(len(sen_samples_predit)):
        sentence_num += 1 
        flag = 1
        for tag in range(len(sen_samples_predit[sentence])):
            tag_num += 1
            if (sen_samples_predit[sentence][tag]) != (sen_samples_true[sentence][tag]):
                flag = 0
                wrong_tag_num += 1 
        if flag == 1:
            correct_sentence_num += 1
            
    tag_accuracy = (tag_num - wrong_tag_num) / tag_num
    sen_accuracy = correct_sentence_num/sentence_num

    return tag_accuracy, sen_accuracy
 

def ouput(sen_samples_predit,tagFreqs_dict):
    index_tag_dict={}
    for tag,index in tagFreqs_dict.items():
        index_tag_dict[index] = tag
    data_path = sys.argv[2]

    test_sample = []
    index = 0 
    with open(data_path) as data:
        for line in data:
            splitline = line.strip().split()
            new_line = ''
            item_index = 0
            for pair in splitline:
                pair = pair.split('/')
                word = pair[0]
                if len(pair) == 2:
                    tag = pair[1]
                    new_line += word+'/'+index_tag_dict[sen_samples_predit[index][item_index+1]]+' '
                    item_index += 1
            index+=1
            test_sample.append(new_line)
    with open('POS.test.out','w') as f :
        for sentenc in test_sample:
            f.write(sentenc)
            f.write('\n')


#   6)  A Simple Baseline Program

def baseline(trainFile, testFile, tag_to_word, tag_to_tag, tagFreqs_dict, wordFreqs_dict):
    tag_count = 0
    accuracy_count = 0
    tag_total = len(tagFreqs_dict)
    word_total = len(wordFreqs_dict)
    word_tag = np.ones((word_total, tag_total))

    with open(trainFile) as data:
        for line in data:
            splitline = line.strip().split()
            new_line = []
            for pair in splitline:
                pair = pair.split('/')
                if len(pair) == 2:
                    new_line.append(pair)
            for i in range(len(new_line)):
                word = new_line[i][0]
                pre_tag = new_line[i][1]
                if word not in wordFreqs_dict:
                    word = 'NON'
                word_tag[wordFreqs_dict[word], tagFreqs_dict[pre_tag]] += 1
    word_tag = word_tag /word_tag.sum(axis=1, keepdims=1)

    with open(testFile) as data:
        for line in data:
            splitline = line.strip().split()
            for pair in splitline:
                pair = pair.split('/')
                if len(pair) == 2:
                    tag_count += 1
                    word = ''
                    if pair[0] not in wordFreqs_dict:
                        word = 'NON'
                    else:
                        word = pair[0]
                    predict_tag = np.argmax(word_tag[wordFreqs_dict[word], :])
                    if predict_tag == tagFreqs_dict[pair[1]]:
                        accuracy_count += 1
    
    
    baseline_accuracy = accuracy_count / tag_count                    
    return baseline_accuracy
    

          
            



if __name__ == '__main__':
    # 1)  Load files
    trainFile = sys.argv[1]
    testFile = sys.argv[2]
    
    # 2)  Process train & test data
    tagFreqs_dict, wordFreqs_dict, train_sample = load_train_data(trainFile)
    
    tag_to_word, tag_to_tag = process_train_data(train_sample,tagFreqs_dict, wordFreqs_dict)
    
    test_sample = load_test_data(testFile)
    
    # 3)  Run Viterbi algorithm 
    sen_samples_predit, sen_samples_true = Viterbi(test_sample,tag_to_word,tag_to_tag, tagFreqs_dict, wordFreqs_dict)
    
    # 4)  Calculate predict score
    tag_accuracy, sen_accuracy  = predict(sen_samples_predit, sen_samples_true)
    
    # 5)  Baseline
    baseline_accuracy = baseline(trainFile, testFile, tag_to_word, tag_to_tag, tagFreqs_dict, wordFreqs_dict)
    
    ouput(sen_samples_predit, tagFreqs_dict)
    
    # print accuracy
    print ("Viterbi tag accuracy is ",tag_accuracy)  
    print('Baseline accuracy is ',baseline_accuracy) 
            
        
    