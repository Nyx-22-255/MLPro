## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro.rl
## -- Module  : models_env.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-04-18  0.0.0     DA       Creation
## -- 2021-06-06  1.0.0     DA       Release of first version
## -- 2021-06-25  1.0.1     DA       New method Environment.get_reward_type();
## -- 2021-08-26  1.1.0     DA       New classes: EnvBase, EnvModel, SARBuffer, SARBufferelement, 
## -- 2021-08-28  1.1.1     DA       Bugfixes and minor improvements
## -- 2021-09-11  1.1.2     MRD      Change Header information to match our new library name
## -- 2021-10-05  1.1.3     DA       Introduction of method Environment.get_cycle_limit()
## -- 2021-10-05  1.1.4     SY       Bugfixes and minor improvements
## -- 2021-10-25  1.1.5     SY       Enhancement of class EnvBase by adding ScientificObject.
## -- 2021-11-26  1.2.0     DA       Redesign:
## --                                - Introduction of special adaptive function classes AFct*
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.0 (2021-11-26)

This module provides model classes for environments and environnment models.
"""


from mlpro.rl.models_sar import *





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AFctSTrans (Model):
    """
    Special adaptive function for state transition prediction.

    Parameters
    ----------
    p_afct_cls : str
        Name of an adaptive function class (compatible to class AdaptiveFunction)
    p_state_space : MSpace
        State space    
    p_action_space: MSpace
        Action space
    p_threshold : float
        See description of class AdaptiveFunction
    p_buffer_size: int
        Initial size of internal data buffer (0=no buffering)
    p_ada : bool
        Boolean switch for adaptivity
    p_logging 
        Log level (see class Log for more details)

    """

    C_TYPE          = 'AFct STrans'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_afct_cls, p_state_space:MSpace, p_action_space:MSpace, p_threshold=0, p_buffer_size=0, p_ada=True, p_logging=Log.C_LOG_ALL):
         
        # concatenate state and action space to input space
        # ...
        input_space = None 

        self._afct = p_afct_cls(p_input_space=input_space, p_output_space=p_state_space, p_output_elem_cls=State, p_threshold=p_threshold, p_buffer_size=p_buffer_size, p_ada=p_ada, p_logging=p_logging)


## -------------------------------------------------------------------------------------------------
    def simulate_reaction(self, p_state:State, p_action:Action) -> State:
        input = None
        return super().map(input)


## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_state:State, p_action:Action) -> bool:
        # to be implemented...
        # 
        #
        pass






## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AFctReward (Model):
    """
    Special adaptive function for reward prediction.

    Parameters
    ----------
    p_afct_cls : str
        Name of an adaptive function class (compatible to class AdaptiveFunction)
    p_state_space : MSpace
        State space    
    p_threshold : float
        See description of class AdaptiveFunction
    p_buffer_size: int
        Initial size of internal data buffer (0=no buffering)
    p_ada : bool
        Boolean switch for adaptivity
    p_logging 
        Log level (see class Log for more details)

    """

    C_TYPE          = 'AFct Reward'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_afct_cls, p_state_space:MSpace, p_threshold=0, p_buffer_size=0, p_ada=True, p_logging=Log.C_LOG_ALL):
        # concatenate state and action space to input space
        # ...
        input_space     = None 
        output_space    = None

        self._afct = p_afct_cls(p_input_space=input_space, p_output_space=output_space, p_output_elem_cls=Element, p_threshold=p_threshold, p_buffer_size=p_buffer_size, p_ada=p_ada, p_logging=p_logging)


## -------------------------------------------------------------------------------------------------
    def compute_reward(self, p_state_old:State, p_state_new:State) -> Reward:
        """
        Predicts the reward based on two consecutive states using the given adaptive function.

        Parameters
        ----------
        p_state_old : State
            State before last action
        p_state_new : State
            State after last action

        Returns
        -------
        Reward
            Object of type Reward
        """

        # to be implemented...
        # 
        #
        pass


## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_state:State, p_action:Action, p_reward:Reward) -> bool:
        """
        Adapts the adaptive function inside.

        Parameters
        ----------
        p_state : State
            State
        p_action : Action
            Action
        p_reward : Reward
            Target value for the reward
        """

        # to be implemented...
        # 
        #
        pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AFctDone (Model):
    """
    Special adaptive function for environment done state prediction.

    Parameters
    ----------
    p_afct_cls : str
        Name of an adaptive function class (compatible to class AdaptiveFunction)
    p_state_space : MSpace
        State space    
    p_threshold : float
        See description of class AdaptiveFunction
    p_buffer_size: int
        Initial size of internal data buffer (0=no buffering)
    p_ada : bool
        Boolean switch for adaptivity
    p_logging 
        Log level (see class Log for more details)

    """

    C_TYPE          = 'AFct Done'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_afct_cls, p_state_space:MSpace, p_threshold=0, p_buffer_size=0, p_ada=True, p_logging=True):
        """
        Parameters:
            p_afct_cls          Name of an adaptive function class (compatible to class AdaptiveFunction)
            p_state_space       State space
            p_threshold         See description of class AdaptiveFunction
            p_buffer_size       Initial size of internal data buffer (0=no buffering)
            p_ada               Boolean switch for adaptivity
            p_logging           Boolean switch for logging functionality
        """
         
        # concatenate state and action space to input space
        # ...
        output_space = None 

        self._afct = p_afct_cls(p_input_space=p_state_space, p_output_space=output_space, p_output_elem_cls=Element, p_threshold=p_threshold, p_buffer_size=p_buffer_size, p_ada=p_ada, p_logging=p_logging)


# -------------------------------------------------------------------------------------------------
    def compute_done(self, p_state:State) -> bool:
        # to be implemented...
        # 
        #
        pass


## -------------------------------------------------------------------------------------------------
    def _adapt(self, *p_args) -> bool:
        # to be implemented...
        # 
        #
        pass




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AFctBroken (Model):
    """
    Special adaptive function for environment broken state prediction.

    Parameters
    ----------
    p_afct_cls : str
        Name of an adaptive function class (compatible to class AdaptiveFunction)
    p_state_space : MSpace
        State space    
    p_threshold : float
        See description of class AdaptiveFunction
    p_buffer_size: int
        Initial size of internal data buffer (0=no buffering)
    p_ada : bool
        Boolean switch for adaptivity
    p_logging 
        Log level (see class Log for more details)

    """

    C_TYPE          = 'AFct Broken'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_afct_cls, p_state_space:MSpace, p_threshold=0, p_buffer_size=0, p_ada=True, p_logging=True):
        """
        Parameters:
            p_afct_cls          Name of an adaptive function class (compatible to class AdaptiveFunction)
            p_state_space       State space
            p_threshold         See description of class AdaptiveFunction
            p_buffer_size       Initial size of internal data buffer (0=no buffering)
            p_ada               Boolean switch for adaptivity
            p_logging           Boolean switch for logging functionality
        """
         
        # concatenate state and action space to input space
        # ...
        output_space = None 

        self._afct = p_afct_cls(p_input_space=p_state_space, p_output_space=output_space, p_output_elem_cls=Element, p_threshold=p_threshold, p_buffer_size=p_buffer_size, p_ada=p_ada, p_logging=p_logging)


## -------------------------------------------------------------------------------------------------
    def compute_broken(self, p_state:State) -> bool:
        # to be implemented...
        # 
        #
        pass


## -------------------------------------------------------------------------------------------------
    def _adapt(self, *p_args) -> bool:
        # to be implemented...
        # 
        #
        pass


       


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class EnvBase (FctReward, FctDone, FctBroken, Plottable, ScientificObject):
    """
    Base class for all environment classes. It defines the interface and elementry properties for
    an environment in the context of reinforcement learning.
    """

    C_TYPE          = 'Environment Base'
    C_NAME          = '????'

    C_LATENCY       = timedelta(0,1,0)              # Default latency 1s

    C_REWARD_TYPE   = Reward.C_TYPE_OVERALL         # Default reward type for reinforcement learning

    C_SCIREF_TYPE   = ScientificObject.C_SCIREF_TYPE_NONE

## -------------------------------------------------------------------------------------------------
    def __init__(self, 
                 p_latency:timedelta=None,      # Optional: latency of environment. If not provided
                                                # internal value C_LATENCY will be used by default.
                 p_logging=Log.C_LOG_ALL):      # Log level (see constants of class Log)

        Log.__init__(self, p_logging=p_logging)
        self._state_space      = ESpace()           # Euclidian space as default
        self._action_space     = ESpace()           # Euclidian space as default
        self._state            = None
        self._last_action      = None
        self._goal_achievement = 0.0
        self.set_latency(p_latency)


## -------------------------------------------------------------------------------------------------
    def get_state_space(self):
        return self._state_space


## -------------------------------------------------------------------------------------------------
    def get_action_space(self):
        return self._action_space


## -------------------------------------------------------------------------------------------------
    def get_latency(self) -> timedelta:
        """
        Returns latency of environment.
        """

        return self._latency


## -------------------------------------------------------------------------------------------------
    def set_latency(self, p_latency:timedelta=None) -> None:
        """
        Sets latency of environment. If p_latency is None latency will be reset
        to internal value of attribute C_LATENCY.

        Parameters:
          p_latency       New latency 
        """

        if p_latency is None:
            self._latency = self.C_LATENCY
        else:
            self._latency = p_latency


## -------------------------------------------------------------------------------------------------
    def get_reward_type(self):
        return self.C_REWARD_TYPE


## -------------------------------------------------------------------------------------------------
    def get_state(self) -> State:
        """
        Returns current state of environment.
        """

        return self._state


## -------------------------------------------------------------------------------------------------
    def _set_state(self, p_state:State) -> None:
        """
        Explicitely sets the current state of the environment. Internal use only.
        """

        self._state = p_state


## -------------------------------------------------------------------------------------------------
    def get_done(self) -> bool:
        if self._state is None: return False
        return self._state.get_done()


## -------------------------------------------------------------------------------------------------
    def get_broken(self) -> bool:
        if self._state is None: return False
        return self._state.get_broken()


## -------------------------------------------------------------------------------------------------
    def get_goal_achievement(self):
        return self._goal_achievement


## -------------------------------------------------------------------------------------------------
    def set_random_seed(self, p_seed=None):
        """
        Resets the internal random generator using the given seed.
        """
        
        random.seed(p_seed)


## -------------------------------------------------------------------------------------------------
    def reset(self, p_seed=None) -> None:
        """
        Resets environment to initial state. Please redefine.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def process_action(self, p_action:Action) -> bool:
        """
        Processes given action and updates the state of the environment.

        Parameters:
            p_action      Action to be processed

        Returns:
            True, if action processing was successfull. False otherwise.
        """

        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Environment (EnvBase, FctSTrans, Mode):
    """
    This class represents the central environment model to be reused/inherited in own rl projects.
    """

    C_TYPE          = 'Environment'
 
    C_CYCLE_LIMIT   = 0             # Recommended cycle limit for training episodes

## -------------------------------------------------------------------------------------------------
    def __init__(self, 
                 p_mode=Mode.C_MODE_SIM,        # Mode of environment (simulation/real)
                 p_latency:timedelta=None,      # Optional: latency of environment. If not provided
                                                # internal value C_LATENCY will be used by default
                 p_logging=Log.C_LOG_ALL):      # Log level (see constants of class Log)

        EnvBase.__init__(self, p_latency, False)
        Mode.__init__(self, p_mode, p_logging)
        self._setup_spaces()


## -------------------------------------------------------------------------------------------------
    def get_cycle_limit(self):
        """
        Returns limit of cycles per training episode.
        """

        return self.C_CYCLE_LIMIT


## -------------------------------------------------------------------------------------------------
    def _setup_spaces(self):
        """
        Implement this method to enrich the state and action space with specific dimensions. 
        """

        # 1 Setup state space
        # self._state_space.add_dim(Dimension(0, 'Pos', 'Position', '', 'm', 'm', [-50,50]))
        # self._state_space.add_dim(Dimension(1, 'Vel', 'Velocity', '', 'm/sec', '\frac{m}{sec}', [-50,50]))

        # 2 Setup action space
        # self.action_space.add_dim(Dimension(0, 'Rot', 'Rotation', '', '1/sec', '\frac{1}{sec}', [-50,50]))

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def process_action(self, p_action:Action) -> bool:
        """
        Processes given action and updates the state of the environment.

        Parameters:
            p_action      Action to be processed

        Returns:
            True, if action processing was successfull. False otherwise.
        """

        # 0 Intro
        self.last_action = p_action
        self.log(self.C_LOG_TYPE_I, 'Start processing action')
        for agent in p_action.get_elem_ids():
            self.log(self.C_LOG_TYPE_I, 'Actions of agent', agent, '=', p_action.get_elem(agent).get_values())


        # 1 State transition
        if self._mode == self.C_MODE_SIM:
            # 1.1 Simulated state transition
            state  = self.get_state()
            new_state = self.simulate_reaction(state, p_action)
            self._set_state(new_state)
            # self._set_state( self.simulate_reaction(self.get_state(), p_action))

        elif self._mode == self.C_MODE_REAL:
            # 1.2 Real state transition

            # 1.2.1 Export action to executing system
            if not self._export_action(p_action):
                self.log(self.C_LOG_TYPE_E, 'Action export failed!')
                return False

            # 1.2.2 Wait for the defined latency
            sleep(self.get_latency().total_seconds())

            # 1.2.3 Import state from executing system
            if not self._import_state():
                self.log(self.C_LOG_TYPE_E, 'State import failed!')
                return False


        # 2 State evaluation
        state = self.get_state()
        state.set_done( self.compute_done(None) )
        state.set_broken( self.compute_broken(None) )
        self._goal_achievement = self._compute_goal_achievement()
        

        # 3 Outro
        self.log(self.C_LOG_TYPE_I, 'Action processing finished successfully')
        return True


## -------------------------------------------------------------------------------------------------
    def _export_action(self, p_action:Action) -> bool:
        """
        Mode C_MODE_REAL only: exports given action to be processed externally 
        (for instance by a real hardware). Please redefine. 

        Parameters:
            p_action      Action to be exported

        Returns:
            True, if action export was successful. False otherwise.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _import_state(self) -> bool:
        """
        Mode C_MODE_REAL only: imports state from an external system (for instance a real hardware). 
        Please redefine. Please use method set_state() for internal update.

        Returns:
          True, if state import was successful. False otherwise.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _compute_goal_achievement(self, p_state:State=None):
        """
        Optional goal achievement computation.

        Parameters:
            p_state         Optional external state. If none, please use internal state.

        Returns:
            Goal avievement value in interval [0,1].
        """

        return 0.0


## -------------------------------------------------------------------------------------------------
    def compute_reward(self, p_state:State=None) -> Reward:
        """
        Computes reward based on given or current state.

        Parameters:
            p_state         State object. If None, please use internal state.

        Returns:
            Reward object
        """

        # if p_state is not None:
        #     state = p_state
        # else:
        #     state = self.get_state()
        
        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class EnvModel(EnvBase, Model):
    """
    Template class for an environment model to be used for model based agents.
    """

    C_TYPE          = 'EnvModel'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_state_space:MSpace, p_action_space:MSpace, p_afct_strans:AFctSTrans, p_afct_reward:AFctReward, p_afct_done:AFctDone, p_afct_broken:AFctBroken, p_ada=True, p_logging=True):
        """
        Parameters:
            p_state_space           State space
            p_action_space          Action space
            p_afct_strans           Adaptive function for state transition prediction
            p_afct_reward           Adaptive function for reward prediction
            p_afct_done             Adaptive function for done prediction
            p_afct_broken           Adaptive function for broken prediction
            p_ada                   Boolean switch for adaptivity
            p_logging               Boolean switch for logging functionality
        """

        EnvBase.__init__(self, p_logging=p_logging)
        Model.__init__(self, p_buffer_size=0, p_ada=p_ada, p_logging=p_logging)
        self._state_space   = p_state_space
        self._action_space  = p_action_space
        self._afct_strans   = p_afct_strans
        self._afct_reward   = p_afct_reward
        self._afct_done     = p_afct_done
        self._afct_broken   = p_afct_broken


## -------------------------------------------------------------------------------------------------
    def process_action(self, p_action: Action) -> bool:
        return super().process_action(p_action)


## -------------------------------------------------------------------------------------------------
    def _adapt(self, *p_args) -> bool:
        """
        Adapts the internal predictive functions based on State-Action-Reward-State (SARS) data.

        Parameters:
            p_arg[0]           Object of type SARSElement
        """

        # ... to be implemented
        pass


## -------------------------------------------------------------------------------------------------
    def get_maturity(self):
        """
        Returns maturity of environment model.
        """

        return min(self._afct_strans.get_maturity(), self._afct_reward.get_maturity(), self._afct_done.get_maturity(), self._afct_broken.get_maturity())


## -------------------------------------------------------------------------------------------------
    def clear_buffer(self):
        self._afct_strans.clear_buffer()
        self._afct_reward.clear_buffer()
        self._afct_done.clear_buffer()
        self._afct_broken.clear_buffer()


## -------------------------------------------------------------------------------------------------
    def get_functions(self):
        return self._afct_strans, self._afct_reward, self._afct_done, self._afct_broken


## -------------------------------------------------------------------------------------------------
    def process_action(self, p_action: Action) -> bool:

        # 1 Concatenate internal state and given action to input element of state transition fct
        # ...
        state_action = None

        # 2 Predict next state
        self._set_state(self._afct_strans.map(state_action))

        return True


## -------------------------------------------------------------------------------------------------
    def _adapt(self, *p_args) -> bool:
        """
        Adapts the environment model based on State-Action-Reward-State (SARS) data.

        Parameters:
            p_arg[0]           Object of type SARSElement
        """

        raise NotImplementedError