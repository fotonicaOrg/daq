# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 14:19:22 2018

@author: Axel Lacapmesure
"""

import numpy as np
from matplotlib import pyplot as plt

data = np.loadtxt('aliasing.txt', delimiter = '\t')
f_in = data[:,0]
f_out = data[:,1]

f_sample = 10000

x = f_in/f_sample
y = f_out/f_sample

plt.plot(
        x,
        y,
        ls = '',
        marker = 'o')
plt.xlabel('$f_{r}/f_{s} $')
plt.ylabel('$f_{m}/f_{s}$')
