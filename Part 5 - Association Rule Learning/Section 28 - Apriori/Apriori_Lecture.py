# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 14:43:10 2020

@author: nsrin
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

transaction = []
for i in range(0,len(dataset)):
    transaction.append( [ str(dataset.values[i,j]) for j in range( len(dataset.values[0]) ) ] ) #Making a 2D list
    
from apyori import apriori
rules = apriori(transaction, min_support = 0.003 ,min_confidence = 0.2 ,min_lift = 3 ,min_length = 2)

results = list(rules)
listed_rules = []
for i in range(0, len(results)):

    listed_rules.append('Rule: ' + str(list(results[i][0])[0]) +

                        ' -> ' + str(list(results[i][0])[1]) +

                        ' S: ' + str(round(results[i].support, 4)) +

                        ' C: ' + str(round(results[i][2][0].confidence, 4)) +

                        ' L: ' + str(round(results[i][2][0].lift, 4)))
#print(results_list)    


