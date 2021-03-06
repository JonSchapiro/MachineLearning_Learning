#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

features_train = features_train[:int(len(features_train)*.01)] 
labels_train = labels_train[:int(len(labels_train)*.01)] 

clf = SVC(C=1000,kernel="rbf")
clf.fit(features_train, labels_train)
preds = clf.predict([features_test[10], features_test[26], features_test[50]])
print(preds)




