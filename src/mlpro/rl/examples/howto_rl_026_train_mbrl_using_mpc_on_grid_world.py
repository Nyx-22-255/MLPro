## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro
## -- Module  : howto_rl_026_train_mbrl_using_mpc_on_grid_world.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-09-19  0.0.0     SY       Creation
## -- 2022-??-??  1.0.0     SY       Release first version
## -------------------------------------------------------------------------------------------------


"""
Ver. 0.0.0 (2022-09-19)

This module shows how to incorporate MPC in Model-Based RL on Grid World problem.

You will learn:
    
1) How to set up an own agent on Grid World problem
    
2) How to set up model-based RL (MBRL) training
    
3) How to incorporate MPC into MBRL training
"""

from mlpro.bf.math import *
from mlpro.rl.models import *
from mlpro.rl.pool.envs.gridworld import *
from mlpro.rl.pool.policies.randomgenerator import RandomGenerator
from pathlib import Path
from mlpro.rl.pool.actionplanner.mpc import MPC
from mlpro.rl.pool.envmodels.mlp_gridworld import MLPEnvModel





# 1 Implement the random RL scenario
class ScenarioGridWorld(RLScenario):

    C_NAME      = 'Grid World with Random Actions'

    def _setup(self, p_mode, p_ada, p_logging):
        # 1.1 Setup environment
        self._env   = GridWorld(p_logging=p_logging,
                                p_action_type=GridWorld.C_ACTION_TYPE_DISC_2D)


        # 1.2 Setup and return random action agent
        policy_random = RandomGenerator(p_observation_space=self._env.get_state_space(), 
                                        p_action_space=self._env.get_action_space(),
                                        p_buffer_size=1,
                                        p_ada=1,
                                        p_logging=p_logging)

        mb_training_param = dict(p_cycle_limit=100,
                                 p_cycles_per_epi_limit=100,
                                 p_max_stagnations=0,
                                 p_collect_states=False,
                                 p_collect_actions=False,
                                 p_collect_rewards=False,
                                 p_collect_training=False)

        return Agent(
            p_policy=policy_random,  
            p_envmodel=MLPEnvModel(),
            p_em_acc_thsld=0.5,
            p_action_planner=MPC(),
            p_predicting_horizon=5,
            p_controlling_horizon=2,
            p_planning_width=5,
            p_name='Smith',
            p_ada=p_ada,
            p_logging=p_logging,
            **mb_training_param
        )



# 2 Create scenario and run the scenario
if __name__ == "__main__":
    # 2.1 Parameters for demo mode
    cycle_limit         = 200
    logging             = Log.C_LOG_ALL
    visualize           = True
    plotting            = True
else:
    # 2.2 Parameters for unittest
    cycle_limit         = 20
    logging             = Log.C_LOG_NOTHING
    visualize           = False
    plotting            = False



# 3 Create your scenario and run some cycles 
myscenario  = ScenarioGridWorld(
    p_mode=Mode.C_MODE_SIM,
    p_ada=True,
    p_cycle_limit=cycle_limit,
    p_visualize=visualize,
    p_logging=logging
)

myscenario.reset(p_seed=3)
myscenario.run() 