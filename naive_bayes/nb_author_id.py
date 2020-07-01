#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""

"""
In this module we learned about Naive Bayes classification.
Naive Bayes uses prior knowledge about the world plus current knowledge to make a posterior claim about the world

Really good explanation of the concept: https://www.youtube.com/watch?v=CPqOCI0ahss
"""
    
import sys
import numpy as np
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

features_train, features_test, labels_train, labels_test = preprocess()
clf = GaussianNB()

# Fit the data using the training data and training labels
clf.fit(features_train, labels_train)

# Make predictions using a subset of the training data
predictions = clf.predict(features_test)

# Check for the accuracy of the predictions by comparing it to the test labels
print(f"Our model uses Naive Bayes to classify an email as belonging to Chris or Sarah by {accuracy_score(predictions, labels_test)}%")



