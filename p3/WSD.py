
# CIS 511 NLP - Assignment 3 - Word Sense Disambiguation

"""
Created on Sat Mar 21 23:31:51 2020

@author: Siyu Yang
@unique name: siyuya
@UMID:76998080

"""

import csv
import math, operator
import sys



#=== Global ===#
FOLDS_NUM = 5
STRIP_LIST= [',', '-', '!','.', '(', ')', '?']

#=== Functions ===#

#   1)  Get Total Number of Instances
def Instances_Number(Filename):
    instances_num = 0
    with open(Filename,'r') as f:
        for line in f:
            if line.find("<instance") > -1:
                instances_num +=1
    return instances_num


#   2)  Parse Training Folds
def Parse_Train(Filename, fold_num):

    train_dict = {}
    train_features_dict = {}
    #get sum of both senses occuring in training set
    total_train_senses = 0  
  
    with open(Filename,'r') as trainfile:
        instance_num = 0
    
        for line in trainfile:
            if line.find("<instance") > -1:
                instance_num +=1
                id_1 = line.split(" ")[1]
                id_2 = id_1.split("=")[1]
                id_3 = id_2.strip("\"")

                sense_1 = next(trainfile)
                sense_2 = sense_1.split(" ")[2]
                sense_3 = sense_2.split('%')[1]
                sense = sense_3.strip("\"/>\n")    
        
                # proper line in training dataset
                if (instance_num-1)%FOLDS_NUM != fold_num:        
          
                    total_train_senses +=1
                    if sense in train_dict:
                        train_dict[sense] += 1
                    else:
                        train_dict[sense] = 1
                        train_features_dict[sense] = {}
       
                    next_line = next(trainfile)
                    next_line = next(trainfile)
          
                    feature_1 = next_line.strip('. \n')
                    features = feature_1.split(" ")
          
                    for word in features:
                        for strips in STRIP_LIST:
                            word = word.strip(strips).lower()
                        word = word.strip()
            
                        if word.find("<head>") == -1:
                            if word not in train_features_dict[sense]:
                                train_features_dict[sense][word] = 1

    return train_dict,train_features_dict,total_train_senses

        

#   2)  Predict Testing Folds
def Predict_Test(Filename, train_dict,train_features_dict,total_train_senses,fold_num,output):

    with open(Filename,'r') as testfile:
        test_num = 0
        correct_num = 0
        instance_num = 0
    
        for line in testfile:
            if line.find("<instance") > -1:
                instance_num +=1      
                id_1 = line.split(" ")[1]
                id_2 = id_1.split("=")[1]
                id_3 = id_2.strip("\"")
        
                sense_1 = next(testfile)
                sense_2 = sense_1.split(" ")[2]
                sense_3 = sense_2.split('%')[1]
                test_sense = sense_3.strip("\"/>\n") 
        
                if (instance_num-1)%FOLDS_NUM == fold_num:
        
                    test_num +=1                  
                    next_line = next(testfile)
                    next_line = next(testfile)
          
                    feature_1 = next_line.strip('. \n')
                    features = feature_1.split(" ")
          
                    for word in features:
                        for strips in STRIP_LIST:
                            word = word.strip(strips).lower()
            
                        word = word.strip()
            
                        if word.find("<head>") == -1:
                            for sense in train_dict:
                                if word not in train_features_dict[sense]:                  
                                    train_features_dict[sense][word] = 0
                  
                    #add-1
                    for sense in train_dict:
                        for word in train_features_dict[sense]:
                            train_features_dict[sense][word] += 0.01
              
                    #calculate the probabilities
                    predict_probs = {}
                    for sense in train_dict:
                        predict_probs[sense] = 1
            
                        for word in features:
                            for strips in STRIP_LIST:
                                word = word.strip(strips)
                            word = word.lower().strip()
              
                            if word.find("<head>") == -1 :
                                predict_probs[sense] = predict_probs[sense] * train_features_dict[sense][word]/train_dict[sense]
                
                        # calculate & factor-in sense probability    
                        predict_probs[sense] = predict_probs[sense] * train_dict[sense]/ total_train_senses
            
                        #print(sense, predict_probs[sense])
                    predict_sense = max(predict_probs.items(), key=operator.itemgetter(1))[0]

                    
                    if predict_sense == test_sense:
                        correct_num +=1
            
                    tag = id_3.split('.')[0]+"%"+predict_sense
                    output.write(id_3 + tag + "\n")

    return correct_num, test_num
 
 

if __name__ == "__main__":

    # 1) load file
    filename = sys.argv[1]
    
    # 2) create outputfile
    outputName = filename + ".out"
    output = open(outputName, "w") 

    
    # 3) Get Total Number of Instances
    instance_count = Instances_Number(filename)

    
    
    # 4) parse and predict
    list_accuracy = []
    
    for fold_num in range(0, FOLDS_NUM):

        
        train_dict,train_features_dict,total_train_senses = Parse_Train(filename, fold_num)
        
        correct_num, test_num = Predict_Test(filename, train_dict,train_features_dict,total_train_senses,fold_num,output)
        
        # 5) calculate accurancy
        accuracy = float(correct_num/test_num)
        print("Fold " + str(fold_num+1) + " Accuracy:", str(accuracy)+"\n")
        
        list_accuracy.append(accuracy)
        
    acc_avg = sum(list_accuracy)/FOLDS_NUM    
    print("Average accuracy:"+ str(acc_avg))   
    output.close()

