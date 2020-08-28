# CIS 511 NLP - Assignment 4.2 - Dialog Act Classification 

"""
Created on Sat Apr 11 15:20:36 2020

@author: Siyu Yang
@unique name: siyuya
@UMID:76998080
"""

import sys
import operator


#=== Functions ===#

#   1)  Pre-processing Functions:

# ### 1) establish possible senses for current word disambiguation

def establish_dialogs(filename):
    
    dialogs_dict = {}
    dialogs_features_dict = {}
    
    with open(filename, encoding='utf-8') as data:

        
        # for each line in the file...
        for line in data:
            
            # if it's the start of a new instance...
            if line.find("Advisor") != -1:
                total_instances += 1
                
                dialog_act = line.split(" ")[1]
                if dialog_act.find('[') == 0:
                    
                    # store or index dialog act (depending on if it's new)
                    if dialog_act not in dialogs_dict:
                        total_dialogs += 1
                        dialogs_dict[dialog_act] = 1
                        dialogs_features_dict[dialog_act] = {}
                    else:
                        dialogs_dict[dialog_act] += 1
        
    return dialogs_dict, dialogs_features_dict, total_instances, total_dialogs


# ### 2) parse the training folds for data counts

def parse_datafile(filename, dialogs_dict, dialogs_features_dict):
    
    stripList = ['.', '(', ')', ',', '-', '!', '?']
    
    with open(filename, encoding='utf-8') as data:

        
        # for each line in the file...
        prev_line = ''
        for line in data:
            
            # if it's the start of a new instance...
            if line.find("Advisor") != -1:
                
                # capture the advisor's dialog act
                dialog_act = line.split(" ")[1]
                if dialog_act.find('[') == 0:
                        
                    # get the features from the previous line
                    features = prev_line.strip('\n')
                    if features.find("Student:") == 0:
                        
                        # loop through each word in the features
                        features_split = features.split(' ')
                        for idx in range(1, len(features_split)):
                            
                            # strip excess puntuation
                            for stripItem in stripList:
                                features_split[idx] = features_split[idx].strip(stripItem)
                            features_split[idx] = features_split[idx].lower().strip()
                        
                            # store all new words as "present"
                            if features_split[idx] not in dialogs_features_dict[dialog_act]:
                                dialogs_features_dict[dialog_act][features_split[idx]] = 1.0
            
            # set previous line for next go-around
            prev_line = line
                                    
    return dialogs_dict, dialogs_features_dict


# ### 3) make predictions on test fold instances


def predict_instances(testFile, outputFile, dialogs_dict, dialogs_features_dict):
    
    stripList = ['.', '(', ')', ',', '-', '!', '?']
    
    with open(testFile) as data:
        test_count = 0
        correct_count = 0

        # for each line in the file...
        prev_line = ''
        for line in data:
            
            # if it's the start of a new instance...
            if line.find("Advisor") != -1:
                test_count += 1
                
                # capture the advisor's dialog act
                dialog_act = line.split(" ")[1]
                true_dialog = dialog_act
                if dialog_act.find('[') == 0:
                        
                    # get the features from the previous line
                    features = prev_line.strip('\n')
                    features_split = []
                    if features.find("Student:") == 0:
                        
                        # loop through each word in the features
                        features_split = features.split(' ')
                        for idx in range(1, len(features_split)):
                            
                            # strip excess puntuation
                            for stripItem in stripList:
                                features_split[idx] = features_split[idx].strip(stripItem)
                            features_split[idx] = features_split[idx].lower().strip()
                        
                            # store all new words as "absent"
                            if features_split[idx] not in dialogs_features_dict[dialog_act]:
                                for dialog in dialogs_dict:
                                    dialogs_features_dict[dialog_act][features_split[idx]] = 0.01
                                
                    
                    # calculate the argmax (probabilities for each dialog)
                    pred_probs = {}
                    for dialog in dialogs_dict:
                        pred_probs[dialog] = 1.0
                        
                        # factor in conditional probability for each word
                        for word in features_split:
                            
                            # use all words except 'Student:' as features
                            if (word != 'Student:'):
                                
                                # store as absent if not already
                                if word not in dialogs_features_dict[dialog]:
                                    dialogs_features_dict[dialog][word] = 0.01
                                    
                                # calculate & factor-in feature probabilities
                                feat_prob = dialogs_features_dict[dialog][word]/dialogs_dict[dialog]
                                pred_probs[dialog] = pred_probs[dialog] * feat_prob
                        
                        # calculate & factor-in dialog probability
                        dialog_prob = dialogs_dict[dialog]/sum(dialogs_dict.values())
                        pred_probs[dialog] = pred_probs[dialog]*dialog_prob
                        #print(dialog, pred_probs[dialog])
                    
                    # identify the dialog with the highest probability
                    pred_dialog = max(pred_probs.items(), key=operator.itemgetter(1))[0]
                    
                    # check if prediction is correct
                    if pred_dialog == true_dialog:
                        correct_count += 1

                    # output prediction
                    outputFile.write(prev_line)
                    outputFile.write(pred_dialog + ' ' + line)
                
            # set previous line for next go-around
            prev_line = line
            
    accuracy = float(correct_count/test_count)
    print('System Accuracy:' + str(accuracy))
    
    return accuracy


if __name__ == "__main__":

    # read in file
    trainFile = sys.argv[1]
    testFile = sys.argv[2]

    # create output file: <word>.wsd.out
    outputName = "DialogAct.test.out"
    outputFile = open(outputName, "w")

    dialogs, features, total_instances, total_dialogs = establish_dialogs(trainFile)


    # ### 4) parse training file for counts
    dialogs_dict, dialogs_features_dict = parse_datafile(trainFile, dialogs, features)


    # ### 5) predict dialog acts for test data
    accuracy = predict_instances(testFile, outputFile, dialogs_dict, dialogs_features_dict)

