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
import nidaqmx.stream_writers
import numpy as np
from matplotlib import pyplot as plt
import time

CAL = 100

ai_channels = ('Dev3/ai0')
co_channels = ('Dev3/ctr0')
pwm_freq = 100
pwm_duty_cycle = 0.9

voltage_range = ([-10,10])
n_samples = 500
freq = 1000

setpoint = 0.35
dt = n_samples/freq
memory_const = 0.25

pid_gain = 1
pid_tau = 340
pid_dead_time = 2

(kp, ki, kd) = daq.tune_cohen_coon(
        controller_type = "pid",
        gain = pid_gain,
        tau = pid_tau,
        dead_time = pid_dead_time
        )
kd = 70

with nidaqmx.Task() as task_ai, nidaqmx.Task() as task_co:
    
    daq.configure_ai(
            task_ai,
            physical_channels = ai_channels,
            voltage_range = voltage_range,
            terminal_configuration = nidaqmx.constants.TerminalConfiguration.RSE
            )
    
    chan_co = daq.configure_pwm(
            task_co,
            physical_channels = co_channels,
            frequency = pwm_freq,
            duty_cycle = pwm_duty_cycle
            )
    
    task_co.timing.cfg_implicit_timing(sample_mode = nidaqmx.constants.AcquisitionType.CONTINUOUS)
    
    stream_co = nidaqmx.stream_writers.CounterWriter(task_co.out_stream)
    task_co.start()
    
    pid = daq.PID_Controller(
        setpoint = setpoint,
        kp = kp,
        ki = ki,
        kd = kd,
        dt = dt
        )
    
    (data, real_freq, control_signal, duty) = daq.continuous_PID(
            pid = pid,
            task_ai = task_ai,
            task_co = task_co,
            stream_co = stream_co,
            chan_co = chan_co[0],
            sample_frequency = freq,
            n_samples = n_samples,
            plot = False
            )

time = np.arange(len(control_signal)) / real_freq * n_samples

plt.subplot(2,1,1)
plt.plot(time, control_signal * daq.CAL, label = "Temperatura")
plt.ylabel("Temperatura [Celsius]")

plt.subplot(2,1,2)
plt.plot(time, duty, label = "Duty cycle")
plt.ylabel("Duty cycle")
plt.xlabel('Tiempo [s]')

#
#plt.plot(time, data[0,:])
#plt.xlabel('Tiempo (s)')
#plt.ylabel('Tensión registrada (V)')
#plt.grid()








