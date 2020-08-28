# CIS 511 NLP - Assignment 4.2 - Dialog Act Classification 

"""
Created on Sat Apr 11 15:20:36 2020

@author: Siyu Yang
@unique name: siyuya
@UMID:76998080
"""

import sys
import math

#=== Functions ===#

#   1)  find the middle content between begin word and end word
def middle_content(begin, end, content):
    mid_content  =''
    if content.find(begin): # find the line start with begin words
        beginword = content[content.find(begin):content.rfind(end)]
        mid_content = beginword[len(begin):]
        return mid_content   
        
      
#   2)  Parse Training File        
def parse_file(filename):

    # dialog_dict: key = DialogAct, value = list of all words 
    dialog_dict = dict() 
    # number_dict: key = DialogAct, value = times in train data
    number_dict = dict() 

    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split("\n")
        for line in lines:
        
            if len(line) > 2:
                prev_words = []
                label_answers = []
                
                if line.startswith("Student:"):
                    words = line.split(" ")
                    for word in words:
                        label_answers.append(word)
                            
                prev_words = label_answers
                
                if line.startswith("Advisor:"):
                    dialog_act = middle_content("[", "]", line)

                    if dialog_act == "social" or dialog_act == "pull" or dialog_act == "push":
                        dialog_act = ""
                    else:
                        if dialog_act not in number_dict:
                            number_dict[dialog_act] = 1
                        else:
                            number_dict[dialog_act] += 1                        
           
                #student's words and advisor's words are both written to
                if len(prev_words) != 0 and len(dialog_act) != 0: 
                    if dialog_act in dialog_dict:
                        for word in prev_words:
                            dialog_dict[dialog_act].append(word)          
                    else:
                        dialog_dict[dialog_act] = prev_words
                    
                    prev_words = []
                    dialog_act = ""                

    return dialog_dict, number_dict


#   3)  Generate Probability Dictionary
def probability(number_dict):
    # prob_dict: key = sense , value = probability of sense
    prob_dict = dict()
    
    total = 0
    for sense in number_dict:
        total += number_dict[sense]
    for sense in number_dict:
        prob_dict[sense] = number_dict[sense] / total

    return prob_dict
    
       
#   4)  Generate Unique Dictionary
def generate_unique_dict(dialog_dict):
    # unique_dict: key = DialogAct, value = list of unique words
    unique_dict = dict()
    
    for dialog_act in dialog_dict:
        unique_dict[dialog_act] = list()
        for word in dialog_dict[dialog_act]:
            if not word in unique_dict[dialog_act]:
                unique_dict[dialog_act].append(word)    
    
    return unique_dict    


#   5)  Process Test Data
def process_test_data(filename):
    # test_dict: key = ID of the line , value = list of all words 
    test_dict = dict() 
    
    ID = 0
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split("\n")
        for line in lines:                            
            if line.startswith("Student:"):
                words = line.split(" ")
                for word in words:
                    if word != "Student:":
                        if ID in test_dict:
                            test_dict[ID].append(word)                        
                        else:
                            test_dict[ID] = [word]
            ID += 1
    return test_dict


#   6)  Add One Smoothing and Get Output
def add_one_smoothing(test_dict, dialog_dict, number_dict, unique_dict, prob_dict, output):

    # score_dict: key = ID , value = sense and score
    score_dict = dict()
    # final_dict: key = ID , value = final label
    final_dict = dict()

    for ID in test_dict:
        for sense in dialog_dict:
            total = 0
            for word in test_dict[ID]:
                # number of word appearance in sense
                num_1 = dialog_dict[sense].count(word) + 1
                # number of sense appearance
                num_2 = number_dict[sense] + len(unique_dict[sense])
                
                total *= math.log((num_1 / num_2), 2)
            
            score = math.log(prob_dict[sense], 2) + total

            if ID not in score_dict:
                score_dict[ID] = list()
            score_dict[ID].append({sense:score})
            
            
    # iterates a dictionary to find argmax and best sense for ID
    final_label = ""
    argMax = -999
    for ID in score_dict:
        for scores in score_dict[ID]:
            for sense in scores:
                score = scores[sense]
                if score > argMax:
                    argMax = score
                    final_label = sense
        final_dict[ID] = final_label

    for ID in final_dict:
        for word in test_dict[ID]:
            output.write(word+" ")
        output.write("\n"+"Label: "+str(final_dict[ID])+"\n\n")        
        
    return final_dict


#   7)  Calculate The Accuracy
def cal_acc(final_dict, testfile):

    # test_out_dict: key = ID , value = test data
    test_out_dict = dict()
    
    with open(testfile, encoding='utf-8' ) as f:
        line_num = 0
        content = f.read()
        lines = content.split("\n")
        for line in lines:
            if len(line) > 2:
                if line.startswith("Advisor:"):
                    dialog_act = middle_content("[", "]", line)            
                if line.startswith("Student:"):
                    if len(dialog_act) > 0:
                        test_out_dict[line_num] = dialog_act
            line_num += 1


    correct_num = 0
    total_num = 0
    for word in final_dict:
        if word in final_dict and word in test_out_dict:
            if final_dict[word] == test_out_dict[word]:
                correct_num += 1
            total_num += 1
    accuracy = correct_num/total_num
    print("Accuracy:", accuracy)
    
    return accuracy





if __name__ == '__main__':

    # 1) load file
    trainFile = sys.argv[1]
    testFile = sys.argv[2]

    # 2) create outputfile
    outputName = testFile + ".out"
    output = open(outputName, "w") 
    
    # 3)  Parse Training File 
    dialog_dict, number_dict = parse_file(trainFile)
    
    # 4)  Generate Probability Dictionary
    prob_dict = probability(number_dict)
    
    # 5)  Generate Unique Dictionary
    unique_dict = generate_unique_dict(dialog_dict)

    # 6)  Process Test Data
    test_dict = process_test_data(testFile)

    # 7)  Add One Smoothing and Get Output
    final_dict = add_one_smoothing(test_dict, dialog_dict, number_dict, unique_dict, prob_dict, output)

    # 8)  Calculate The Accuracy    
    cal_acc(final_dict, testFile)
