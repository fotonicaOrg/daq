# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 16:11:37 2018

@author: Guillermo
"""


import numpy as np
from matplotlib import pyplot as plt

T1 = control_signal*100

d1 = duty

t1 = time


plt.plot(t1,T1,'ob',markersize=1,label='Filtro PI')
plt.plot(t1,T1/T1*35,'k--',color =(0.6,0.6,0.6),label='Temperatura de referencia')
plt.legend(loc=4)
plt.xlabel('Tiempo (s)')
plt.ylabel('Temperatura (°C)')