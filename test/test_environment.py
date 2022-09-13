## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro
## -- Module  : test_environment
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-09-11  1.0.0     MRD      Creation
## -- 2021-09-11  1.0.0     MRD      Release First Version
## -- 2021-09-22  1.0.1     WB       Change Environment Instantiation Method
## -- 2021-09-26  1.0.2     MRD      Change the structure to work with GitHub Automated Test
## -- 2021-09-26  1.0.3     MRD      Change the import module due to the change of the pool
## --                                folder structer
## -- 2021-09-26  1.0.4     MRD      Change the import module due to the change of the pool
## --                                folder structer
## -- 2022-09-02  1.0.5     SY       Add DoublePendulumS7 and DoublePendulumS4
## -- 2022-09-13  1.0.5     SY       Add Sim_MPPS
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.5 (2022-09-13)

Unit test classes for environment.
"""


import pytest
import random
import numpy as np
from mlpro.rl.models import *
from mlpro.rl.pool.envs.robotinhtm import RobotHTM
from mlpro.rl.pool.envs.bglp import BGLP
from mlpro.rl.pool.envs.gridworld import GridWorld
from mlpro.rl.pool.envs.multicartpole import MultiCartPole
from mlpro.rl.pool.envs.doublependulum import DoublePendulumS7
from mlpro.rl.pool.envs.doublependulum import DoublePendulumS4
# from mlpro.rl.pool.envs.ur5jointcontrol import UR5JointControl
from mlpro.rl.pool.envs.mpps import Sim_MPPS


## -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("env_cls", [RobotHTM, BGLP, GridWorld, MultiCartPole, DoublePendulumS7, DoublePendulumS4,
                                     Sim_MPPS])
def test_environment(env_cls):
    env = env_cls()
    assert isinstance(env, Environment)
    
    assert isinstance(env.get_state_space(), ESpace)
    assert env.get_state_space().get_num_dim() != 0
    
    assert isinstance(env.get_action_space(), ESpace)
    assert env.get_action_space().get_num_dim() != 0
    
    state = env.get_state()
    
    assert isinstance(state, State)
        
    my_action_values = np.zeros(env.get_action_space().get_num_dim())
    for d in range(env.get_action_space().get_num_dim()):
        my_action_values[d] = random.random() 

    my_action_values = Action(0, env.get_action_space(), my_action_values)

    env.process_action(my_action_values)

    reward = env.compute_reward()
    
    assert isinstance(reward, Reward)

    env.reset()
