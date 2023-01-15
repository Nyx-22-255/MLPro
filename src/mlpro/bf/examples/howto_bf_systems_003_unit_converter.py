## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.examples
## -- Module  : howto_bf_systems_003_unit_converter.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-01-15  0.0.0     SY       Creation
## -- 2023-01-15  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-01-15)

This module provides an example of using the unit converter in MLPro.

You will learn:

1) How to use the the unit converter

"""


from mlpro.bf.systems import UnitConverter
from mlpro.bf.various import Log



if __name__ == "__main__":
    p_print = True
else:
    p_print = False

# 1 Initialize unit converters
conv_length = UnitConverter(p_name='conv_length',
                            p_type=UnitConverter.C_UNIT_CONV_LENGTH,
                            p_unit_in='m',
                            p_unit_out='km')

conv_pressure = UnitConverter(p_name='conv_pressure',
                              p_type=UnitConverter.C_UNIT_CONV_PRESSURE,
                              p_unit_in='bar',
                              p_unit_out='Pa')

conv_current = UnitConverter(p_name='conv_current',
                             p_type=UnitConverter.C_UNIT_CONV_CURRENT,
                             p_unit_in='mA',
                             p_unit_out='A')

conv_force = UnitConverter(p_name='conv_force',
                           p_type=UnitConverter.C_UNIT_CONV_FORCE,
                           p_unit_in='N',
                           p_unit_out='J/cm')

conv_power = UnitConverter(p_name='conv_power',
                           p_type=UnitConverter.C_UNIT_CONV_POWER,
                           p_unit_in='W',
                           p_unit_out='kW')

conv_mass = UnitConverter(p_name='conv_mass',
                          p_type=UnitConverter.C_UNIT_CONV_MASS,
                          p_unit_in='kg',
                          p_unit_out='lb')

conv_time = UnitConverter(p_name='conv_time',
                          p_type=UnitConverter.C_UNIT_CONV_TIME,
                          p_unit_in='hr',
                          p_unit_out='s')

conv_temperature = UnitConverter(p_name='conv_temperature',
                                 p_type=UnitConverter.C_UNIT_CONV_TEMPERATURE,
                                 p_unit_in='K',
                                 p_unit_out='F')


# 2. Call the defined unit converters
p_input = 10
output = conv_length.call(p_input)
if p_print:
    print('We convert %.1f%s to %.2f%s'%(p_input, conv_length._unit_in, output, conv_length._unit_out))
    
p_input = 10
output = conv_pressure.call(p_input)
if p_print:
    print('We convert %.1f%s to %.1f%s'%(p_input, conv_pressure._unit_in, output, conv_pressure._unit_out))
    
p_input = 10
output = conv_current.call(p_input)
if p_print:
    print('We convert %.1f%s to %.2f%s'%(p_input, conv_current._unit_in, output, conv_current._unit_out))
    
p_input = 10
output = conv_force.call(p_input)
if p_print:
    print('We convert %.1f%s to %.1f%s'%(p_input, conv_force._unit_in, output, conv_force._unit_out))
    
p_input = 10
output = conv_power.call(p_input)
if p_print:
    print('We convert %.1f%s to %.2f%s'%(p_input, conv_power._unit_in, output, conv_power._unit_out))
    
p_input = 10
output = conv_mass.call(p_input)
if p_print:
    print('We convert %.1f%s to %.3f%s'%(p_input, conv_mass._unit_in, output, conv_mass._unit_out))
    
p_input = 10
output = conv_time.call(p_input)
if p_print:
    print('We convert %.1f%s to %.1f%s'%(p_input, conv_time._unit_in, output, conv_time._unit_out))
    
p_input = 10
output = conv_temperature.call(p_input)
if p_print:
    print('We convert %.1f%s to %.2f%s'%(p_input, conv_temperature._unit_in, output, conv_temperature._unit_out))