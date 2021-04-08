# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Fillig Missing Values

from sklearn.impute import SimpleImputer
imputer_2 = SimpleImputer(missing_values=np.nan,strategy='mean') #New method
imputer_2 = imputer_2.fit(x[:,1:3])
x[:,1:3]=imputer_2.transform(x[:,1:3])

#categorical Values Encoding

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_encoder_country = LabelEncoder()
x[:,0] = label_encoder_country.fit_transform(x[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
x = onehotencoder.fit_transform(x).toarray()

label_encoder_purchased = LabelEncoder()
y = label_encoder_purchased.fit_transform(y)

#Spliting

from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#Scaling

from sklearn.preprocessing import StandardScaler
sc_x= StandardScaler()
x_train= sc_x.fit_transform(x_train)
x_test= sc_x.transform(x_test)





