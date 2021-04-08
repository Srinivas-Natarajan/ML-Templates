# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 13:29:17 2020

@author: nsrin
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:26:08 2020

@author: nsrin
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

N=10000                    # Number of data items
d=10                        #No of ads
reward_1 = [0] *d 
reward_0 = [0] *d
ad_selected = []
total_reward = 0

for n in range(N):
    ad = 0
    max_random = 0
   
    for i in range(d):
        random_beta = random.betavariate(reward_1[i] + 1, reward_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ad_selected.append(ad)
    reward= dataset.values[n,ad]
    if reward == 1:
        reward_1[ad] += 1
    else:
        reward_0[ad] += 1
    total_reward = total_reward + reward

plt.hist(ad_selected)
plt.title('Histogram of Ad Selection')
plt.xlabel('Ads')
plt.ylabel('No of Selections')
        