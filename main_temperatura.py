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

CAL = 100

channels = ('Dev2/ai0')
voltage_range = ([-10,10])
n_samples = 10
freq = 100

with nidaqmx.Task() as task:        
    (data, real_freq) = daq.continuous_acquire(
                task,
                n_samples,
                channels,
                freq,
                voltage_range = voltage_range,
                acquisition_units = nidaqmx.constants.VoltageUnits.VOLTS,
                terminal_configuration = nidaqmx.constants.TerminalConfiguration.RSE
                )

time = np.arange(data.size) / real_freq

plt.plot(time, data[0,:])
plt.xlabel('Tiempo (s)')
plt.ylabel('Tensión registrada (V)')
plt.grid()








