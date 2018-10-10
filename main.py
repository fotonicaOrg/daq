# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 14:58:55 2018

@author: Axel Lacapmesure
"""

import sys
if "daq" not in sys.modules:
    import daq
else:
    import importlib.reload
    importlib.reload(daq)

import nidaqmx
import numpy as np
from matplotlib import pyplot as plt

frequencies = np.linspace(1e3, 125e3, 20)
channels = ('Dev2/ai0','Dev2/ai8')
voltage_range = ([-10,10], [-0.2, 0.2])
n_samples = 1000000
real_frequencies = []

mean_5v = []
mean_0v = []
std_5v = []
std_0v = []

for freq in frequencies:

    with nidaqmx.Task() as task:    
        
        (data, real_freq) = daq.acquire(
            task,
            n_samples,
            channels,
            freq,
            voltage_range = voltage_range,
            acquisition_units = nidaqmx.constants.VoltageUnits.VOLTS,
            terminal_configuration = nidaqmx.constants.TerminalConfiguration.RSE
            )
        
        real_frequencies.append(real_freq)
    
    mean_5v.append(np.mean(data[0,:]))
    mean_0v.append(np.mean(data[1,:]))
    std_5v.append(np.std(data[0,:]))
    std_0v.append(np.std(data[1,:]))
    
mean_5v = np.array(mean_5v)
mean_0v = np.array(mean_0v)
std_5v = np.array(std_5v)
std_0v = np.array(std_0v)

plt.plot(frequencies/1000, mean_0v*1e6)
plt.xlabel('Frecuencia de muestreo (kHz)')
plt.ylabel('Tensión registrada (uV)')
plt.grid()

plt.figure()
plt.plot(frequencies/1000, (mean_5v-mean_5v[0])*1e6)
plt.xlabel('Frecuencia de muestreo (kHz)')
plt.ylabel('Tensión registrada (uV)')
plt.grid()








