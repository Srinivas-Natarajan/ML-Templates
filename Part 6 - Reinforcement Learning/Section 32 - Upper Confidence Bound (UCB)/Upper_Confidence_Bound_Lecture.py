# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:26:08 2020

@author: nsrin
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

N=10000                    # Number of data items
d=10                        #No of ads
sum_of_rewards = [0] * d
no_of_selections = [0] * d
ad_selected = []

for n in range(N):
    ad=0
    max_upper_bound = 0
   
    for i in range(d):
        if(no_of_selections[i] > 0):
            avg_reward = sum_of_rewards[i]/no_of_selections[i]
            delta = math.sqrt( 3/2 * math.log(n+1) / no_of_selections[i] )
            upper_bound = avg_reward + delta
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ad_selected.append(ad)
    no_of_selections[ad]+=1
    reward= dataset.values[n,ad]
    sum_of_rewards[ad] += reward


plt.hist(ad_selected)
plt.title('Histogram of Ad Selection')
plt.xlabel('Ads')
plt.ylabel('No of Selections')
        