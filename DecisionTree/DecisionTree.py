# CIS 511 NLP - Final Project

"""
Created on Sat April 18 20:30:18 2020

@author: Siyu Yang
@unique name: siyuya
@UMID:76998080
"""

from collections import defaultdict
import pandas
from sklearn.tree import DecisionTreeClassifier                
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import re
import sys


#=== Functions ===#


#   1)  Process Train File to get train data        
def Process_TrainFile(trainfile):
    
    trainList = open(trainfile,'r',encoding='utf-8',errors='ignore').read() #read the file and remove the empty lines.
    train_fea_dic = defaultdict(list) # use default dictionary as train file feature dictionary

    for line in trainList.split("\n\n"):
    
        if('**********' in line):
            text = line.split("\n")
            text1 = text[0]
            result = text[-1].split(" ")[-1]
            text2 = re.sub('([.,!?()])', r' \1 ', text1)  
            text3 = re.sub('\s{2,}', ' ', text2)    
            value= text3.split(" ") #get the value from each line
            
            while "" in value: #remove empty lines
                value.remove("")
                
            for i in range(0, len(value)):
                # Feature 1:  the value of the token 
                train_fea_dic['Value'].append(value[i])

                # Feature 2:  is token all uppercase?
                if(value[i].isupper):
                    train_fea_dic['UpperCase'].append(1)
                else:
                    train_fea_dic['UpperCase'].append(0)

                # Feature 3:  does token start with capital?  
                if(value[i].istitle()):
                    train_fea_dic['Capital'].append(1)
                else:
                    train_fea_dic['Capital'].append(0)
                
                # Feature 4:  length of token 
                train_fea_dic['Length'].append(len(value[i]))
                
                # Feature 5:  length of left word < 4  
                if(len(value[i])<4):
                    train_fea_dic['Length<Four'].append(1)
                else:
                    train_fea_dic['Length<Four'].append(0)                
                
                # Feature 6:  if Right word contains "!"
                if('!' in value[i]):
                    train_fea_dic['Exclamation_in_token'].append(1) 
                else:
                    train_fea_dic['Exclamation_in_token'].append(0)
                    
                # Feature 7:  if Right word contains ":"
                if(':' in value[i]):
                    train_fea_dic['Colon_in_token'].append(1) 
                else:
                    train_fea_dic['Colon_in_token'].append(0)
                
                # Feature 8:  results "po/nt/ng"               
                if(result == 'po'):
                    train_fea_dic['Result'].append(2)
                elif(result =='nt'):
                    train_fea_dic['Result'].append(1)
                else:
                    train_fea_dic['Result'].append(0)


    train_data = pandas.DataFrame.from_dict(train_fea_dic)
    train_data['Value'] = train_data.index    

    
    return train_data


#   2)  Process Test File to get test data
def Process_TestFile(testfile):            
    testList = open(testfile,'r',encoding='utf-8',errors='ignore').read() #read the file and remove the empty lines.
    test_fea_dic = defaultdict(list) # use default dictionary as test file feature dictionary

    for line in testList.split("\n\n"):       
        if("**********" in line):
            testtext = line.split("\n")      
            testtext1 = testtext[0]
            testresult = testtext[-1].split(" ")[-1]
            testtext2 = re.sub('([.,!?()])', r' \1 ', testtext1)  
            testtext3 = re.sub('\s{2,}', ' ', testtext2)    
            testvalue= testtext3.split(" ") #get the value from each line
            
            while "" in testvalue: #remove empty lines
                testvalue.remove("")

            for i in range(0, len(testvalue)):
                # Feature 1:  the value of the token 
                test_fea_dic['Value'].append(testvalue[i])

                # Feature 2:  is token all uppercase?
                if(testvalue[i].isupper):
                    test_fea_dic['UpperCase'].append(1)
                else:
                    test_fea_dic['UpperCase'].append(0)

                # Feature 3:  does token start with capital?  
                if(testvalue[i].istitle()):
                    test_fea_dic['Capital'].append(1)
                else:
                    test_fea_dic['Capital'].append(0)
                
                # Feature 4:  length of token 
                test_fea_dic['Length'].append(len(testvalue[i]))

                    
                # Feature 5:  length of left word < 4    
                if(len(testvalue[i])<4):
                    test_fea_dic['Length<Four'].append(1)
                else:
                    test_fea_dic['Length<Four'].append(0)
                                        
                # Feature 6:  if Right word contains "!"
                if('!' in testvalue[i]):
                    test_fea_dic['Exclamation_in_token'].append(1) 
                else:
                    test_fea_dic['Exclamation_in_token'].append(0)

                # Feature 7:  if Right word contains ":"
                if(':' in testvalue[i]):
                    test_fea_dic['Colon_in_token'].append(1) 
                else:
                    test_fea_dic['Colon_in_token'].append(0)

                # Feature 8:  results "po/nt/ng"                      
                if(testresult == 'po'):
                    test_fea_dic['Result'].append(2)
                elif(testresult =='nt'):
                    test_fea_dic['Result'].append(1)
                else:
                    test_fea_dic['Result'].append(0)  
  
    test_data = pandas.DataFrame.from_dict(test_fea_dic)
    test_data['Value'] = test_data.index    

    return test_data,test_fea_dic
 
 
#   3)  Calculate The Accuracy
def accuracy(train_data, test_data,test_fea_dic):  
 
    features = ['Value','UpperCase','Capital','Length','Length<Four','Exclamation_in_token', 'Colon_in_token']
    X = train_data[features] 
    Y = train_data['Result']

    test_X = test_data[features]
    test_Y = test_data['Result']
    X_train,  X_test, Y_train, Y_test = train_test_split(X, Y)

    # Create Decision Tree classifer object and get the predict value
    predict = DecisionTreeClassifier().fit(X_train,Y_train).predict(test_X)

    # Calculate and print the accuracy score    
    accuracy = accuracy_score(test_Y,predict)
    print("Accuracy:",str(accuracy*100) + '%')
    
    return accuracy, predict




#   4)  Precision: TP/(TP+FP)
def precision(testfile, predict):

    counter = 0
    poTP = 0
    poFP = 0
    ntTP = 0
    ntFP = 0
    ngTP = 0
    ngFP = 0

    testList = open(testfile,'r',encoding='utf-8',errors='ignore').read() #read the file and remove the empty lines.
    test_fea_dic = defaultdict(list) # use default dictionary as test file feature dictionary

    for line in testList.split("\n\n"):       
        if("**********" in line):
            testtext = line.split("\n")      
            testtext1 = testtext[0]
            testresult = testtext[-1].split(" ")[-1]
            testtext2 = re.sub('([.,!?()])', r' \1 ', testtext1)  
            testtext3 = re.sub('\s{2,}', ' ', testtext2)    
            testvalue= testtext3.split(" ") #get the value from each line
            
            
            while "" in testvalue: #remove empty lines
                testvalue.remove("")
            for i in range(0, len(testvalue)):  
               
                if predict[counter] == 2:
                    if (testresult == 'po'):
                        poTP += 1
                    else:
                        poFP += 1
                if predict[counter] == 1:
                    if (testresult == 'nt'):
                        ntTP += 1
                    else:
                        ntFP += 1
                if predict[counter] == 0:
                    if (testresult == 'ng'):
                        ngTP += 1
                    else:
                        ngFP += 1
                counter += 1
            

    if (poTP+poFP) == 0:
        print('Precision of positive reviews: N/A (zero count)')
        poPrecision = 0
    else:
        print('Precision of positive reviews: ' + str(round(poTP / (poTP+poFP) * 100, 2)) + '%')
        poPrecision = poTP / (poTP+poFP)
    
    if (ntTP+ntFP) == 0:
        print('Precision of neutral reviews: N/A (zero count)')
        ntPrecision = 0
    else:
        print('Precision of neutral reviews: ' + str(round(ntTP / (ntTP+ntFP) * 100, 2)) + '%')
        ntPrecision = ntTP / (ntTP+ntFP)
    
    if (ngTP+ngFP) == 0:
        print('Precision of negative reviews: N/A (zero count)')
        ngPrecision = 0
    else:
        print('Precision of negative reviews: ' + str(round(ngTP / (ngTP+ngFP) * 100, 2)) + '%')
        ngPrecision = ngTP / (ngTP+ngFP)

    return poPrecision, ntPrecision, ngPrecision
        


#   5)  Recall: TP/(TP+FN)
def recall(testfile, predict):
    counter = 0
    poTP = 0
    poFP = 0
    poFN = 0
    ntTP = 0
    ntFP = 0
    ntFN = 0
    ngTP = 0
    ngFP = 0
    ngFN = 0

    testList = open(testfile,'r',encoding='utf-8',errors='ignore').read() #read the file and remove the empty lines.
    test_fea_dic = defaultdict(list) # use default dictionary as test file feature dictionary

    for line in testList.split("\n\n"):       
        if("**********" in line):
            testtext = line.split("\n")      
            testtext1 = testtext[0]
            testresult = testtext[-1].split(" ")[-1]
            testtext2 = re.sub('([.,!?()])', r' \1 ', testtext1)  
            testtext3 = re.sub('\s{2,}', ' ', testtext2)    
            testvalue= testtext3.split(" ") #get the value from each line
            
            
            while "" in testvalue: #remove empty lines
                testvalue.remove("")
            for i in range(0, len(testvalue)):  

                if testresult == 'po':
                    if (predict[counter] == 2):
                        poTP += 1
                    else:
                        poFN += 1
                if testresult == 'nt':
                    if (predict[counter] == 1):
                        ntTP += 1
                    else:
                        ntFN += 1
                if testresult == 'ng':
                    if (predict[counter] == 0):
                        ngTP += 1
                    else:
                        ngFN += 1
                counter += 1


    if (poTP+poFN) == 0:
        print('Recall of positive reviews: N/A (zero count)')
        poRecall = 0
    else:
        print('Recall of positive reviews: ' + str(round(poTP / (poTP+poFN) * 100, 2)) + '%')
        poRecall = poTP / (poTP+poFN)
    if (ntTP+ntFN) == 0:
        print('Recall of neutral reviews: N/A (zero count)')
        ntRecall = 0
    else:
        print('Recall of neutral reviews: ' + str(round(ntTP / (ntTP+ntFN) * 100, 2)) + '%')
        ntRecall = ntTP / (ntTP+ntFN)
    if (ngTP+ngFN) == 0:
        print('Recall of negative reviews: N/A (zero count)')
        ngRecall = 0
    else:
        print('Recall of negative reviews: ' + str(round(ngTP / (ngTP+ngFN) * 100, 2)) + '%')
        ngRecall = ngTP / (ngTP+ngFN)

    return poRecall, ntRecall, ngRecall 



#   6)  F1-Score: 2 × (precision × recall)/(precision + recall)
def f1score(poPrecision, ntPrecision, ngPrecision, poRecall, ntRecall, ngRecall):

    if (poRecall+poPrecision) == 0:
        print('F-1 score of positive reviews: N/A (zero count)')
    else:
        print('F-1 score of positive reviews: ' + str(round(2 * ((poRecall*poPrecision)/(poRecall+poPrecision)) * 100, 2)) + '%')
    if (ntRecall+ntPrecision) == 0:
        print('F-1 score of neutral reviews: N/A (zero count)')
    else:
        print('F-1 score of neutral reviews: ' + str(round(2 * ((ntRecall*ntPrecision)/(ntRecall+ntPrecision)) * 100, 2)) + '%')
    if (ngRecall+ngPrecision) == 0:
        print('F-1 score of negative reviews: N/A (zero count)')
    else:
        print('F-1 score of negative reviews: ' + str(round(2 * ((ngRecall*ngPrecision)/(ngRecall+ngPrecision)) * 100, 2)) + '%')
    return 0


  

  
if __name__ == "__main__":

    # 1) load file
    trainfile = sys.argv[1]
    testfile = sys.argv[2]
    
    # 2)  Process Train File to get train data    
    train_data= Process_TrainFile(trainfile)
    
    # 3)  Process Test File to get test data    
    test_data,test_fea_dic = Process_TestFile(testfile)
    
    # 4) generate output and calculate accurancy
    accuracy, predict = accuracy(train_data, test_data,test_fea_dic)      

    # 5) Precision
    poPrecision, ntPrecision, ngPrecision = precision(testfile, predict)  

    # 6) Recall
    poRecall, ntRecall, ngRecall = recall(testfile, predict)  

    # 7) F1-Score
    f1score(poPrecision, ntPrecision, ngPrecision, poRecall, ntRecall, ngRecall) 


