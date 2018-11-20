# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 11:01:15 2018

@author: Axel Lacapmesure
"""
from matplotlib import pyplot as plt
import numpy as np

ax1 = plt.subplot(2,1,1)
plt.plot(time, control_signal * daq.CAL)
plt.ylabel("Temperatura [°C]")
plt.setp(ax1.get_xticklabels(), visible=False)

ax2 = plt.gcf().add_axes([.37, .63, .5, .17], zorder = 100)
plt.plot(time, control_signal * daq.CAL, label = "Temperatura")
plt.xlim([175,325])
plt.ylim([49.8, 50.2])
plt.xticks([175,250,325])
plt.yticks([49.8,50,50.2])

ax3 = plt.subplot(2,1,2, sharex = ax1)
plt.plot(time, duty, label = "Duty cycle")
plt.ylabel("Duty cycle")
plt.xlabel('Tiempo [s]')
plt.tight_layout()
