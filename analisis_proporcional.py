# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 15:38:17 2018

@author: Guillermo
"""

import numpy as np
from matplotlib import pyplot as plt

T1 = control_signal*100
T2 = control_signal000*100
T3 = control_signal001*100

d1 = duty
d2 = duty000
d3 = duty001



t1 = time
t2 = time000
t3 = time001


plt.plot(t1,T1,'ob',markersize=1,label='kp = 40')
plt.plot(t2,T2,'or',markersize=1,label='kp = 60')
plt.plot(t3,T3,'og',markersize=1,label='kp = 200')
plt.plot(t1,T1/T1*35,'k--',color =(0.6,0.6,0.6),label='Temperatura de referencia')
plt.legend(loc=4)
plt.xlabel('Tiempo (s)')
plt.ylabel('Temperatura (°C)')