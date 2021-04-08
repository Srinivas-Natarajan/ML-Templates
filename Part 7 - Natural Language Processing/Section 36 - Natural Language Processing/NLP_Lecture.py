# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 15:50:55 2020

@author: nsrin
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('stopwords')
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

corpus = []

for i in range(1000):
    review = re.sub('[^a-zA-z]',' ',dataset['Review'][i]) #Remove Non-Alphabetical Characters
    review = review.lower()                     #Lower case
    review = review.split()                     #Split words into list
    ps = PorterStemmer()                        #Replacing words with its source/stem from which it is from
    review = [ ps.stem(word) for word in review if word not in stopwords.words('english') ] #Replace words by their root
    review = ' '.join(review)
    corpus.append(review)



cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()                               #Naive Bayes Model
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



