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

def check_device():
    
    import nidaqmx.system
    
    system = nidaqmx.system.System.local()
    
    for dev in system.devices:
        print(dev)

def acquire(
        task,
        n_samples,
        physical_channels,
        sample_frequency,
        voltage_range,
        acquisition_units = nidaqmx.constants.VoltageUnits.VOLTS,
        terminal_configuration = nidaqmx.constants.TerminalConfiguration.RSE
        ):
    
    if not isinstance(voltage_range, tuple):
        voltage_range = [voltage_range]
    
    if not isinstance(physical_channels, tuple):
        physical_channels = [physical_channels]
    
    chan = []
    
    for i in range(len(physical_channels)):
        chan.append(
                task.ai_channels.add_ai_voltage_chan(
                    physical_channel = physical_channels[i],
                    min_val = voltage_range[i][0],
                    max_val = voltage_range[i][1],
                    units = acquisition_units
                    )
                )
            
        chan[i].ai_term_cfg = terminal_configuration
    
    task.timing.cfg_samp_clk_timing(sample_frequency)
    
    data = np.zeros((task.number_of_channels, n_samples))
    reader = nidaqmx.stream_readers.AnalogMultiChannelReader(task.in_stream)
    reader.read_many_sample(data, -1)
    
    return (data, task.timing.samp_clk_rate)





def continuous_acquire(
        task,
        n_samples,
        physical_channels,
        sample_frequency,
        voltage_range,
        acquisition_units = nidaqmx.constants.VoltageUnits.VOLTS,
        terminal_configuration = nidaqmx.constants.TerminalConfiguration.RSE,
        ):
    
    data_count = 0
    
    if not isinstance(voltage_range, tuple):
        voltage_range = [voltage_range]
    
    if not isinstance(physical_channels, tuple):
        physical_channels = [physical_channels]
    
    chan = []
    
    for i in range(len(physical_channels)):
        chan.append(
                task.ai_channels.add_ai_voltage_chan(
                    physical_channel = physical_channels[i],
                    min_val = voltage_range[i][0],
                    max_val = voltage_range[i][1],
                    units = acquisition_units
                    )
                )
            
        chan[i].ai_term_cfg = terminal_configuration
    
    n_channels = task.number_of_channels
    
    task.timing.cfg_samp_clk_timing(rate = sample_frequency,
                                    sample_mode = nidaqmx.constants.AcquisitionType.CONTINUOUS)
    task.in_stream.input_buf_size = 5 * n_samples
    
    data = np.zeros((n_channels, n_samples))
    
    try:
        
        task.start()
        
        while True:
            
            data[0:n_channels, 0:n_samples] = np.array(task.read(n_samples))
            data_count += n_samples
        
    except KeyboardInterrupt:
        
        task.stop()
        
        return (data, task.timing.samp_clk_rate)


if __name__ is '__main__':
    
    check_device()
    
#
#SampleNumber = 2**16
#SampleRate = 10000
#
#with nidaqmx.Task() as task:
#    
#    
#    
#    
#    
#      
##data = data.transpose()    
#data = data[0,:]
#data_fft = 2*np.fft.fft(data)/len(data)
#
#data_fft = np.abs(data_fft[0:len(data_fft)//2])
#freq = np.linspace(0,SampleRate/2,len(data_fft))
#f_medida = freq[np.argmax(data_fft)]
#
#print(f_medida)
#
#
#plt.plot(freq,data_fft)
#
#
#
#    
#    
#    
#    
#    
#    