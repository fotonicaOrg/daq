# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 14:57:30 2018

@author: Guillermo
"""

import numpy as np
from matplotlib import pyplot as plt

## Cargar los archivos en orden alfa ascendente

T1 = duty*100
T2 = duty000*100
T3 = duty001*100
T4 = duty002*100

d1 = control_signal
d2 = control_signal000
d3 = control_signal001
d4 = control_signal002


t1 = time
t2 = time000
t3 = time001
t4 = time002

plt.subplot(4,1,1)
plt.plot(t1,T1,'g',label='a = 0.2')
plt.legend(loc=4)

plt.xlim(0,600)

plt.subplot(4,1,2)
plt.plot(t2,T2,'b',label='a = 2')
plt.legend(loc=4)
plt.xlim(0,600)

plt.subplot(4,1,3)
plt.plot(t3,T3,'r',label='a = 20')
plt.legend(loc=4)
plt.xlim(0,600)

plt.subplot(4,1,4)
plt.plot(t4,T4,'k',label='a = 200')
plt.legend(loc=4)
plt.ylabel('Temperatura (°C)')
plt.xlabel('Tiempo (s)')
plt.xlim(0,600)

#plt.subplot(4,1,1)
#plt.plot(t1,d1,'g',label='a = 0.2')
#plt.legend(loc=1)
#
#plt.xlim(0,600)
#
#plt.subplot(4,1,2)
#plt.plot(t2,d2,'b',label='a = 2')
#plt.legend(loc=1)
#plt.xlim(0,600)
#
#plt.subplot(4,1,3)
#plt.plot(t3,d3,'r',label='a = 20')
#plt.legend(loc=1)
#plt.xlim(0,600)
#
#plt.subplot(4,1,4)
#plt.plot(t4,d4,'k',label='a = 200')
#plt.legend(loc=1)
#plt.ylabel('Ancho de pulso PWM')
#plt.xlabel('Tiempo (s)')
#plt.xlim(0,600)