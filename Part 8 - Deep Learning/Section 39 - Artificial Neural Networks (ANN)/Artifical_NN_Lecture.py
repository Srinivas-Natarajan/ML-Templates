# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 15:56:47 2020

@author: nsrin
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:, -1].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


import keras
from keras.layers import Dense
from keras.models import Sequential

classifier = Sequential()   #init replaced by kernal_initializer
classifier.add( Dense(6, input_dim = 11, kernel_initializer = 'uniform', activation = 'relu') ) #First Hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))    #Second Hidden Layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid')) #Output Layer
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)
#'adam' = Stochastic Gradient Function


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)  #Thresholding with limit 0.5. True if >0.5, False if <.5


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
