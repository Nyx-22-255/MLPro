## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl.envs
## -- Module  : mpps.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-08-24  0.0.0     SY/ML    Creation
## -- 2022-??-??  1.0.0     SY/ML    Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2022-08-24)

This module provides a multi-purpose environment of a continuous and batch production systems with
modular settings and high-flexibility.

The users are able to develop and simulate their own production systems including setting up own
actuators, reservoirs, modules/stations, production sequences and many more. We also provide the
default implementations of actuators, reservoirs, and modules, which can be found in the pool of
objects.

To be noted, the usage of this simulation is not limited to RL tasks, but it also can be as a
testing environment for GT tasks, evolutionary algorithms, supervised learning, model predictive
control, and many more.
"""


from mlpro.rl.models import *
from mlpro.bf.various import *
import numpy as np
import random




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

class Actuator:


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_name, **p_param):
        ...


## -------------------------------------------------------------------------------------------------
    def set_id(self):
        ...


## -------------------------------------------------------------------------------------------------
    def get_id(self):
        ...


## -------------------------------------------------------------------------------------------------
    def set_name(self):
        ...


## -------------------------------------------------------------------------------------------------
    def get_name(self):
        ...
        

## -------------------------------------------------------------------------------------------------
    def activate(self):
        ...


## -------------------------------------------------------------------------------------------------    
    def deactivate(self):
        ...
  
    
## -------------------------------------------------------------------------------------------------      
    def get_status(self):
        ... #on/off


## -------------------------------------------------------------------------------------------------        
    def reset(self):
        ...


## -------------------------------------------------------------------------------------------------    
    def emergency_stop(self):
        ...


## -------------------------------------------------------------------------------------------------    
    def force_stop(self):
        ...


## -------------------------------------------------------------------------------------------------    
    def setup_process(self):
        ...


## -------------------------------------------------------------------------------------------------    
    def run_process(self):
        ...




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

class Reservoir:


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_name, p_max_capacity, p_sensor, **p_param):
        ... # sensor: continuous values or 2-postion sensors (low,mid,high)
        ...


## -------------------------------------------------------------------------------------------------
    def set_id(self):
        ...


## -------------------------------------------------------------------------------------------------
    def get_id(self):
        ...


## -------------------------------------------------------------------------------------------------
    def set_name(self):
        ...


## -------------------------------------------------------------------------------------------------
    def get_name(self):
        ...
        

## -------------------------------------------------------------------------------------------------
    def update(self, p_in, p_out):
        ...
        

## -------------------------------------------------------------------------------------------------
    def get_volume(self):
        ... # according to sensor type
        

## -------------------------------------------------------------------------------------------------
    def get_maximum_capacity(self):
        ...
        

## -------------------------------------------------------------------------------------------------
    def get_overflow(self):
        ...
        

## -------------------------------------------------------------------------------------------------
    def reset(self):
        ...
        

## -------------------------------------------------------------------------------------------------
    def get_initial_level(self):
        ...




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

class Module:


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_name, **p_param):
        ...


## -------------------------------------------------------------------------------------------------
    def set_id(self):
        ...


## -------------------------------------------------------------------------------------------------
    def get_id(self):
        ...


## -------------------------------------------------------------------------------------------------
    def set_name(self):
        ...


## -------------------------------------------------------------------------------------------------
    def get_name(self):
        ...


## -------------------------------------------------------------------------------------------------
    def add_actuator(self, p_pos):
        ...


## -------------------------------------------------------------------------------------------------
    def add_reservoir(self, p_pos):
        ...


## -------------------------------------------------------------------------------------------------
    def setup_sequence(self):
        ...


## -------------------------------------------------------------------------------------------------
    def get_information(self):
        ... # information regarding the actuators, reservoirs, and sequences


## -------------------------------------------------------------------------------------------------
    def reset(self):
        ...




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

class TransferFunction:


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_name, p_type, **p_param):
        ...


## -------------------------------------------------------------------------------------------------
    def set_id(self):
        ...


## -------------------------------------------------------------------------------------------------
    def get_id(self):
        ...


## -------------------------------------------------------------------------------------------------
    def set_name(self):
        ...


## -------------------------------------------------------------------------------------------------
    def get_name(self):
        ...


## -------------------------------------------------------------------------------------------------
    def setup_function(self):
        ...


## -------------------------------------------------------------------------------------------------
    def function_approximation(self, **p_args):
        ...


## -------------------------------------------------------------------------------------------------
    def call(self, p_input):
        ...


## -------------------------------------------------------------------------------------------------
    def plot(self, p_window, p_input_min, p_input_max):
        ...




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

class Process:


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_name, **p_param):
        ...


## -------------------------------------------------------------------------------------------------
    def add(self, **p_args):
        ...
        # self.all_processes.append(TransferFunction(.....))


## -------------------------------------------------------------------------------------------------
    def run(self):
        ...




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

class Sim_MPPS(Environment):


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_name, **p_param):
        ...


## -------------------------------------------------------------------------------------------------
    def set_id(self):
        ...


## -------------------------------------------------------------------------------------------------
    def get_id(self):
        ...


## -------------------------------------------------------------------------------------------------
    def set_name(self):
        ...


## -------------------------------------------------------------------------------------------------
    def get_name(self):
        ...


## -------------------------------------------------------------------------------------------------
    def add_module(self):
        ...


## -------------------------------------------------------------------------------------------------
    def setup_modules(self):
        ...


## -------------------------------------------------------------------------------------------------
    def to_be_added(self):
        ... # to be added later
        





        



    
    
