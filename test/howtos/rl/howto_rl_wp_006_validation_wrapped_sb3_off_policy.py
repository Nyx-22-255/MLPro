## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl.examples
## -- Module  : howto_rl_wp_006_validation_wrapped_sb3_off_policy.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-01-11  0.0.0     MRD      Creation
## -- 2022-01-18  1.0.0     MRD      Released first version
## -- 2022-02-27  1.0.1     SY       Refactoring due to auto generated ID in class Dimension
## -- 2022-07-20  1.0.2     SY       Update due to the latest introduction of Gym 0.25
## -- 2022-10-14  1.0.3     SY       Refactoring 
## -- 2022-11-07  1.1.0     DA       Refactoring 
## -- 2023-01-14  1.1.1     MRD      Removing default parameter new_step_api and render_mode for gym
## -- 2023-02-02  1.2.0     DA       Refactoring 
## -- 2023-02-04  1.2.1     SY       Refactoring to avoid printing during unit test
## -- 2023-02-13  1.2.2     DA       Optimization of dark mode
## -- 2023-03-27  1.3.0     DA       Refactoring
## -- 2023-04-19  1.3.1     MRD      Refactor module import gym to gymnasium
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.3.1 (2023-04-19)

This module shows comparison between native and wrapped SB3 policy (Off-policy).
"""


import gymnasium as gym
import pandas as pd
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from mlpro.bf.plot import DataPlotting
from mlpro.rl import *
from mlpro.wrappers.gymnasium import WrEnvGYM2MLPro
from mlpro.wrappers.sb3 import WrPolicySB32MLPro
from pathlib import Path



# 1 Parameter
if __name__ == "__main__":
    # 1.1 Parameters for demo mode
    logging = Log.C_LOG_NOTHING
    visualize = False
    path = str(Path.home())
    cycle_limit = 600

else:
    # 1.2 Parameters for internal unit test
    logging = Log.C_LOG_NOTHING
    visualize = False
    path = None
    cycle_limit = 10

mva_window = 1
buffer_size = 100
policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=[10])


# 2 Implement your own RL scenario
class MyScenario(RLScenario):
    C_NAME = 'Howto-RL-WP-006'

    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging) -> Model:
        class CustomWrapperFixedSeed(WrEnvGYM2MLPro):
            def _reset(self, p_seed=None):
                self.log(self.C_LOG_TYPE_I, 'Reset')
                self._num_cycles = 0

                # 1 Reset Gym environment and determine initial state
                observation, _ = self._gym_env.reset()
                obs = DataObject(observation)

                # 2 Create state object from Gym observation
                state = State(self._state_space)
                state.set_values(obs.get_data())
                self._set_state(state)
                
        # 1 Setup environment
        if p_visualize:
            gym_env     = gym.make('CartPole-v1', render_mode="human")
        else:
            gym_env     = gym.make('CartPole-v1')
        self._env = CustomWrapperFixedSeed(gym_env, p_seed=2, p_logging=p_logging)

        # 2 Instatiate Policy From SB3
        # DQN
        policy_sb3 = DQN(
            policy="MlpPolicy",
            learning_starts=12,
            buffer_size=24,
            env=None,
            _init_setup_model=False,
            policy_kwargs=policy_kwargs,
            device="cpu",
            seed=2)

        # 3 Wrap the policy
        self.policy_wrapped = WrPolicySB32MLPro(
            p_sb3_policy=policy_sb3,
            p_cycle_limit=self._cycle_limit,
            p_observation_space=self._env.get_state_space(),
            p_action_space=self._env.get_action_space(),
            p_ada=p_ada,
            p_visualize=p_visualize,
            p_logging=p_logging)

        # 4 Setup standard single-agent with own policy
        return Agent(
            p_policy=self.policy_wrapped,
            p_envmodel=None,
            p_name='Smith',
            p_ada=p_ada,
            p_visualize=p_visualize,
            p_logging=p_logging
        )


# 3 Instantiate training
training = RLTraining(
    p_scenario_cls=MyScenario,
    p_cycle_limit=cycle_limit,
    p_collect_states=True,
    p_collect_actions=True,
    p_collect_rewards=True,
    p_collect_eval=True,
    p_path=path,
    p_visualize=visualize,
    p_logging=logging)


# 4 Train SB3 Wrapper
training.run()


# 5 Create Plotting Class
class MyDataPlotting(DataPlotting):
    def get_plots(self):
        """
        A function to plot data
        """
        for name in self.data.names:
            maxval = 0
            minval = 0
            if self.printing[name][0]:
                fig = plt.figure(figsize=(7, 7))
                raw = []
                label = []
                ax = fig.subplots(1, 1)
                ax.set_title(name)
                ax.grid(True, which="both", axis="both")
                for fr_id in self.data.frame_id[name]:
                    raw.append(np.sum(self.data.get_values(name, fr_id)))
                    if self.printing[name][1] == -1:
                        maxval = max(raw)
                        minval = min(raw)
                    else:
                        maxval = self.printing[name][2]
                        minval = self.printing[name][1]

                    label.append("%s" % fr_id)
                ax.plot(raw)
                ax.set_ylim(minval - (abs(minval) * 0.1), maxval + (maxval * 0.1))
                ax.set_xlabel("Episode")
                ax.legend(label, bbox_to_anchor=(1, 0.5), loc="center left")
                self.plots[0].append(name)
                self.plots[1].append(ax)
                if self.showing:
                    plt.show()
                else:
                    plt.close(fig)


# 6 Plotting 1 MLpro
data_printing = {"Cycle": [False],
                 "Day": [False],
                 "Second": [False],
                 "Microsecond": [False],
                 training.get_scenario().get_model().get_name(): [True, -1]}

mem = training.get_results().ds_rewards
mem_plot = MyDataPlotting(mem, p_showing=False, p_printing=data_printing)
mem_plot.get_plots()
wrapper_plot = mem_plot.plots


# 7 Create Callback for the SB3 Training
class CustomCallback(BaseCallback, Log):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    C_TYPE = 'Wrapper'
    C_NAME = 'SB3 Policy'

    def __init__(self, p_verbose=0, p_logging=False):
        Log.__init__(self, p_logging=p_logging)
        super(CustomCallback, self).__init__(p_verbose)
        reward_space = Set()
        reward_space.add_dim(Dimension("Native"))
        self.ds_rewards = RLDataStoring(reward_space)
        self.episode_num = 0
        self.total_cycle = 0
        self.cycles = 0
        self.plots = None
        self.new_episodes = False
        Log.__init__(self, p_logging=p_logging)

        self.continue_training = True
        self.rewards_cnt = []

    def _on_training_start(self) -> None:
        self.log(self.C_LOG_TYPE_I, Training.C_LOG_SEPARATOR)
        self.log(self.C_LOG_TYPE_I, '-- Episode', self.episode_num, 'started...')
        self.log(self.C_LOG_TYPE_I, Training.C_LOG_SEPARATOR, '\n')
        self.ds_rewards.add_episode(self.episode_num)

    def _on_step(self) -> bool:
        if self.new_episodes:
            self.log(self.C_LOG_TYPE_I, Training.C_LOG_SEPARATOR)
            self.log(self.C_LOG_TYPE_I, '-- Episode', self.episode_num, 'finished after', self.total_cycle, 'cycles')
            self.log(self.C_LOG_TYPE_I, Training.C_LOG_SEPARATOR, '\n\n')
            self.episode_num += 1
            self.total_cycle = 0
            self.ds_rewards.add_episode(self.episode_num)
            self.log(self.C_LOG_TYPE_I, Training.C_LOG_SEPARATOR)
            self.log(self.C_LOG_TYPE_I, '-- Episode', self.episode_num, 'started...')
            self.log(self.C_LOG_TYPE_I, Training.C_LOG_SEPARATOR, '\n')
            self.new_episodes = False
        # With Cycle Limit
        self.ds_rewards.memorize_row(self.total_cycle, timedelta(0, 0, 0), self.locals.get("rewards"))
        self.total_cycle += 1
        self.cycles += 1
        if self.locals.get("done"):
            self.new_episodes = True

        return True

    def _on_training_end(self) -> None:
        self.log(self.C_LOG_TYPE_I, 'Training cycle limit', self.cycles, 'reached')
        data_printing = {"Cycle": [False],
                         "Day": [False],
                         "Second": [False],
                         "Microsecond": [False],
                         "Native": [True, -1]}
        mem_plot = MyDataPlotting(self.ds_rewards, p_showing=False, p_printing=data_printing)
        mem_plot.get_plots()
        self.plots = mem_plot.plots


# 8 Run the SB3 Training Native
gym_env = gym.make('CartPole-v1')
policy_sb3 = DQN(
    policy="MlpPolicy",
    learning_starts=12,
    buffer_size=24,
    env=gym_env,
    policy_kwargs=policy_kwargs,
    device="cpu",
    seed=2)

cus_callback = CustomCallback(p_logging=logging)
policy_sb3.learn(total_timesteps=1200, callback=cus_callback)
native_plot = cus_callback.plots


# 9 Difference Plot
native_ydata = native_plot[1][0].lines[0].get_ydata()
wrapper_ydata = wrapper_plot[1][0].lines[0].get_ydata()
smoothed_native = pd.Series.rolling(pd.Series(native_ydata), mva_window).mean()
smoothed_native = [elem for elem in smoothed_native]
smoothed_wrapper = pd.Series.rolling(pd.Series(wrapper_ydata), mva_window).mean()
smoothed_wrapper = [elem for elem in smoothed_wrapper]
plt.plot(smoothed_native, label="Native")
plt.plot(smoothed_wrapper, label="Wrapper")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()

if __name__ == "__main__":
    plt.show()
