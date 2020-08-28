# CIS 511 NLP - Assignment 4.1 - Natural Language Understanding for Dialog Systems 

"""
Created on Sat April 11 20:30:18 2020

@author: Siyu Yang
@unique name: siyuya
@UMID:76998080

"""

from collections import defaultdict
import pandas as p
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import re
import sys


    



with open(r'C:\Users\Yang Siyu\Desktop\p4\NLU.train','r')as f:
    nlupara = [line.rstrip('\n') for line in f]
    print (nlupara)
    