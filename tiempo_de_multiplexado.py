# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 17:53:40 2018

@author: Axel Lacapmesure
"""

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

channels = ('Dev2/ai0','Dev2/ai1','Dev2/ai2')
fs = 120e3/len(channels)
voltage_range = ([-10,10],[-10,10],[-10, 10])
n_samples = 2

with nidaqmx.Task() as task:    
    
    (data, real_freq) = daq.acquire(
        task,
        n_samples,
        channels,
        fs,
        voltage_range = voltage_range,
        acquisition_units = nidaqmx.constants.VoltageUnits.VOLTS,
        terminal_configuration = nidaqmx.constants.TerminalConfiguration.DIFFERENTIAL
        )

time = 1000 * np.arange(data.shape[1]) / real_freq

mean = np.mean(data, 1)
sort_idx = np.argsort(mean)

tau = np.mean(np.diff(mean[sort_idx]))

print("Tau = {:0.4g} us".format(tau * (1/fs)/6))

plt.figure()
plt.plot(time, data[2,:], '.-', label = channels[2])
plt.plot(time, data[1,:], '.-', label = channels[1])
plt.plot(time, data[0,:], '.-', label = channels[0])
plt.xlabel('Tiempo (ms)')
plt.ylabel('Tensión registrada (V)')
plt.grid()
plt.legend()








