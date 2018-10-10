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

plt.plot(
        f_in,
        f_out,
        label = 'Mediciones',
        ls = '',
        marker = 'o')
plt.xlabel('Frecuencia real [Hz]')
plt.ylabel('Frecuencia adquirida [Hz]')
plt.legend()