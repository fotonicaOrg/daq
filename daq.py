# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import nidaqmx
import collections
from matplotlib import pyplot as plt
import nidaqmx.stream_readers
import nidaqmx.system
#system = nidaqmx.system.System.local()
#
#for device in system.devices:
#    print(device)


SampleNumber = 2**16
SampleRate = 10000

with nidaqmx.Task() as task:
    chan1 = task.ai_channels.add_ai_voltage_chan(
            physical_channel = 'Dev1/ai0',
            min_val = -5,
            max_val = 5,
            units = nidaqmx.constants.VoltageUnits.VOLTS)
#    chan2 = task.ai_channels.add_ai_voltage_chan(
#            physical_channel = 'Dev1/ai1',
#            min_val = -5,
#            max_val = 5,
#            units = nidaqmx.constants.VoltageUnits.VOLTS)
    
    task.timing.cfg_samp_clk_timing(SampleRate)
    chan1.ai_term_cfg = nidaqmx.constants.TerminalConfiguration.RSE
#    chan2.ai_term_cfg = nidaqmx.constants.TerminalConfiguration.RSE
    
    data = np.zeros((task.number_of_channels,SampleNumber))
    reader = nidaqmx.stream_readers.AnalogMultiChannelReader(task.in_stream)
    reader.read_many_sample(data,SampleNumber)
    
    
      
#data = data.transpose()    
data = data[0,:]
data_fft = 2*np.fft.fft(data)/len(data)

data_fft = np.abs(data_fft[0:len(data_fft)//2])
freq = np.linspace(0,SampleRate/2,len(data_fft))
f_medida = freq[np.argmax(data_fft)]

print(f_medida)


plt.plot(freq,data_fft)



    
    
    
    
    
    