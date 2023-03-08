## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl.examples
## -- Module  : howto_rl_mb_003_train_and_reload_model_based_agent_gym.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-03-07  1.0.0     DA       Adapted from howto_rl_agent_011
## -- 2023-03-08  1.0.1     SY       Refactoring and quality assurance
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.1 (2023-03-08)

This module shows how to train a single agent in MBRL and load it again to do some extra cycles.

You will learn:

1) How to use the RLScenario class of MLPro.

2) How to save a scenario after some run.

3) How to reload the saved scenario and re-run for additional cycles.

"""


import gym
import torch
from stable_baselines3 import PPO
from mlpro.rl import *
from mlpro.wrappers.openai_gym import WrEnvGYM2MLPro
from mlpro.wrappers.sb3 import WrPolicySB32MLPro
from mlpro.sl.pool.afct.fnn.pytorch.mlp import PyTorchMLP
from pathlib import Path
              

            
                
                
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

# 1 Set up MLP for Grid World
class GymCartPoleAFct(PyTorchMLP):
    
    C_NAME = "Cart Pole Adaptive Function"
    
    def _init_hyperparam(self, **p_par): 
        self._hyperparam_space.add_dim(HyperParam('input_size','Z'))
        self._hyperparam_space.add_dim(HyperParam('output_size','Z'))
        self._hyperparam_space.add_dim(HyperParam('update_rate','Z'))
        self._hyperparam_space.add_dim(HyperParam('num_hidden_layers','Z'))
        self._hyperparam_space.add_dim(HyperParam('hidden_size','Z'))
        self._hyperparam_space.add_dim(HyperParam('activation_fct'))
        self._hyperparam_space.add_dim(HyperParam('output_activation_fct'))
        self._hyperparam_space.add_dim(HyperParam('optimizer'))
        self._hyperparam_space.add_dim(HyperParam('loss_fct'))
        self._hyperparam_space.add_dim(HyperParam('learning_rate','R'))
        self._hyperparam_tuple = HyperParamTuple(self._hyperparam_space)
        
        ids_ = self._hyperparam_tuple.get_dim_ids()
        self._hyperparam_tuple.set_value(ids_[0], self._input_space.get_num_dim())
        self._hyperparam_tuple.set_value(ids_[1], self._output_space.get_num_dim())
        self._hyperparam_tuple.set_value(ids_[2], p_par['p_update_rate'])
        self._hyperparam_tuple.set_value(ids_[3], p_par['p_num_hidden_layers'])
        self._hyperparam_tuple.set_value(ids_[4], p_par['p_hidden_size'])
        self._hyperparam_tuple.set_value(ids_[5], p_par['p_act_fct'])
        self._hyperparam_tuple.set_value(ids_[6], p_par['p_output_afc_fct'])
        self._hyperparam_tuple.set_value(ids_[7], p_par['p_optimizer'])
        self._hyperparam_tuple.set_value(ids_[8], p_par['p_loss_fct'])
        self._hyperparam_tuple.set_value(ids_[9], p_par['p_learning_rate'])
                      

                    
                        
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class CartPoleEnvModel(EnvModel):


## -------------------------------------------------------------------------------------------------
    def _compute_reward(self, p_state_old: State = None, p_state_new: State = None) -> Reward:
        reward = Reward(Reward.C_TYPE_OVERALL)
        reward.set_overall_reward(0)
        return reward
                      

                    
                        
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

# 2 Implement your own RL scenario
class MyScenario (RLScenario):
    
    C_NAME = 'Matrix'


## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging) -> Model:

        # 2.1 Setup environment
        gym_env = gym.make('CartPole-v1')
        self._env = WrEnvGYM2MLPro(gym_env, p_visualize=p_visualize, p_logging=p_logging)
        self._env.reset()

        # 2.2 Setup Policy from SB3
        policy_sb3 = PPO(
            policy="MlpPolicy",
            n_steps=10,
            env=None,
            _init_setup_model=False,
            device="cpu",
            seed=1)

        # 2.3 Wrap the policy
        policy_wrapped = WrPolicySB32MLPro(
            p_sb3_policy=policy_sb3,
            p_cycle_limit=self._cycle_limit,
            p_observation_space=self._env.get_state_space(),
            p_action_space=self._env.get_action_space(),
            p_ada=p_ada,
            p_visualize=p_visualize,
            p_logging=p_logging)
        

        # 2.4 Setup a model-based agent

        # 2.4.1 Setup adaptive function for state transition
        afct_strans = AFctSTrans( p_afct_cls=GymCartPoleAFct,
                                  p_state_space=self._env.get_state_space(),
                                  p_action_space=self._env.get_action_space(),
                                  p_threshold=1.8,
                                  p_buffer_size=5000,
                                  p_ada=p_ada,
                                  p_visualize=p_visualize,
                                  p_logging=p_logging,
                                  p_update_rate=1,
                                  p_num_hidden_layers=3,
                                  p_hidden_size=128,
                                  p_act_fct=torch.nn.ReLU,
                                  p_output_afc_fct=torch.nn.ReLU,
                                  p_optimizer=torch.optim.Adam,
                                  p_loss_fct=torch.nn.MSELoss,
                                  p_learning_rate=3e-4 )

        # 2.4.2 Setup environment model
        envmodel = CartPoleEnvModel( p_observation_space=self._env.get_state_space(),
                                    p_action_space=self._env.get_action_space(),
                                    p_latency=self._env.get_latency(),
                                    p_afct_strans=afct_strans,
                                    p_afct_reward=None,
                                    p_afct_success=self._env,
                                    p_afct_broken=self._env,
                                    p_ada=p_ada,
                                    p_init_states=self._env.get_state(),
                                    p_visualize=p_visualize,
                                    p_logging=p_logging )
        

        mb_training_param = dict( p_cycle_limit=100,
                                  p_cycles_per_epi_limit=100,
                                  p_max_stagnations=0,
                                  p_collect_states=False,
                                  p_collect_actions=False,
                                  p_collect_rewards=False,
                                  p_collect_training=False )

        # 2.4.3 Setup standard single-agent with sb3 policy and environment model
        return Agent(
            p_policy=policy_wrapped,
            p_envmodel=envmodel,
            p_name='Smith',
            p_ada=p_ada,
            p_visualize=p_visualize,
            p_logging=p_logging,
            **mb_training_param
        )
                      

                    
                        
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    # Parameters for demo mode
    cycle_limit = 10000
    adaptation_limit = 0
    stagnation_limit = 0
    eval_frequency = 0
    eval_grp_size = 0
    logging = Log.C_LOG_WE
    visualize = True
    path = str(Path.home())

else:
    # Parameters for internal unit test
    cycle_limit = 10
    adaptation_limit = 5
    stagnation_limit = 5
    eval_frequency = 2
    eval_grp_size = 1
    logging = Log.C_LOG_NOTHING
    visualize = False
    path = str(Path.home())


# 3 Create scenario and start training
training = RLTraining(
    p_scenario_cls=MyScenario,
    p_cycle_limit=cycle_limit,
    p_adaptation_limit=adaptation_limit,
    p_stagnation_limit=stagnation_limit,
    p_eval_frequency=eval_frequency,
    p_eval_grp_size=eval_grp_size,
    p_path=path,
    p_visualize=visualize,
    p_logging=logging )



# 4 Training
training.run()



# 5 Reload the scenario
if __name__ == '__main__':
    input( '\nTraining finished. Press ENTER to reload and run the scenario...\n')

scenario = MyScenario.load( p_path = training.get_training_path() + os.sep + 'scenario' )


# 6 Reset Scenario
scenario.reset()  


# 7 Run Scenario
scenario.run()

if __name__ != '__main__':
    from shutil import rmtree
    rmtree(training.get_training_path())
else:
    input( '\nPress ENTER to finish...')
