## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.oa.systems
## -- Module  : systems.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-mm-mm  0.0.0     LSB      Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2023-02-16)

This module provides modules and template classes for adaptive systems and adaptive functions.
"""





from mlpro.bf.ml.systems import *
from mlpro.bf.systems import *
from mlpro.bf.ml import Model
from mlpro.bf.streams import *
from mlpro.oa.streams import *
from typing import Callable, Dict






## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PseudoTask(OATask):




## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_wrap_method:Callable[[List[Instance],
                                         List[Instance]],
                                         None],
                 p_name='PseudoTask',
                 p_range_max=Range.C_RANGE_NONE,
                 p_duplicate_data=True,
                 p_logging=Log.C_LOG_ALL,
                 p_visualize=False,
                 **p_kwargs):


        OATask.__init__(self,
                        p_name = p_name,
                        p_range_max = p_range_max,
                        p_duplicate_data = p_duplicate_data,
                        p_ada = False,
                        p_logging = p_logging,
                        p_visualize= p_visualize,
                        **p_kwargs)


        self._host_task = p_wrap_method


## -------------------------------------------------------------------------------------------------
    def _run( self,
              p_inst_new : list,
              p_inst_del : list ):

        self._host_task(p_inst_new = p_inst_new,
                        p_inst_del = p_inst_del)






## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
# class OAFctBase():
#
#
# ## -------------------------------------------------------------------------------------------------
#     def __init__(self, p_workflow, p_wrap_method):
#
#
#         self._processing_wf: OAWorkflow = p_workflow
#         self._wrap_method = p_wrap_method
#         self._first_fct_run:bool = True
#
#
# ## -------------------------------------------------------------------------------------------------
#     def _setup_fct_workflow(self):
#         """
#
#         Returns
#         -------
#
#         """
#
#         self._processing_wf.add_task(PseudoTask(p_wrap_method = self._wrap_method))
#         return False
#

# ## -------------------------------------------------------------------------------------------------
#     def add_task(self, p_task : StreamTask, p_pred_tasks: list = None):
#         """
#
#         Parameters
#         ----------
#         p_task
#         p_pred_tasks
#
#         Returns
#         -------
#
#         """
#
#         if p_task is not None:
#             self._processing_wf.add_task(p_task = p_task, p_pred_tasks=p_pred_tasks)
#
#
# ## -------------------------------------------------------------------------------------------------
#     def run( self,
#              p_range : int = None,
#              p_wait: bool = False,
#              p_inst_new : list = None,
#              p_inst_del : list = None ):
#         """
#
#         Parameters
#         ----------
#         p_range
#         p_wait
#         p_inst_new
#         p_inst_del
#
#         Returns
#         -------
#
#         """
#
#         if self._first_fct_run:
#             self._first_fct_run = self._setup_fct_workflow()
#
#         self._processing_wf.run(p_range = p_range,
#                                 p_wait = p_wait,
#                                 p_inst_new=p_inst_new,
#                                 p_inst_del=p_inst_del)







## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAFctSTrans(FctSTrans, Model):
    """

    Parameters
    ----------
    p_name
    p_range_max
    p_class_shared
    p_visualize
    p_logging
    p_kwargs
    """

    def __init__(self,
                 p_afct_cls = None,
                 p_state_space: MSpace = None,
                 p_action_space: MSpace = None,
                 p_input_space_cls=ESpace,
                 p_output_space_cls=ESpace,
                 p_output_elem_cls=State,  # Specific output element type
                 p_threshold=0,
                 p_buffer_size=0,
                 p_ada:bool=True,
                 # p_visualize:bool=False,
                 # p_logging=Log.C_LOG_ALL,
                 # p_name:str=None,
                 # p_range_max=Async.C_RANGE_THREAD,
                 # p_class_shared=None,
                 p_wf: OAWorkflow = None,
                 p_visualize:bool=False,
                 p_logging=Log.C_LOG_ALL,
                 **p_kwargs):

        self._afct_strans = None
        if p_afct_cls is not None:
            if (p_state_space is None) or (p_action_space is None):
                raise ParamError("Please provide mandatory parameters state and action space.")

            self._afct_strans = AFctSTrans(p_afct_cls = p_afct_cls,
                                          p_state_space=p_state_space,
                                          p_action_space=p_action_space,
                                          p_input_space_cls=p_input_space_cls,
                                          p_output_space_cls=p_output_space_cls,
                                          p_output_elem_cls=p_output_elem_cls,
                                          p_threshold=p_threshold,
                                          p_buffer_size=p_buffer_size,
                                          p_ada=p_ada,
                                          p_visualize=p_visualize,
                                          p_logging=p_logging,
                                          **p_kwargs)
        #
        # else:
        FctSTrans.__init__(self, p_logging = p_logging)

        Model.__init__(self)
        # OAFctBase.__init__(self)
        self._wf = p_wf
        # self._strans_task:StreamTask = None
        self._instance: Instance = None
        # self._shared = p_class_shared
        self._state:State = None
        self._action:Action = None
    ## -------------------------------------------------------------------------------------------------


## -------------------------------------------------------------------------------------------------
    def simulate_reaction(self, p_state: State, p_action: Action, p_t_step : timedelta = None) -> State:
        """

        Parameters
        ----------
        p_state
        p_action

        Returns
        -------

        """
        self.log(Log.C_LOG_TYPE_I, 'Reaction Simulation Started...')


        # 1. check if the user has already created a workflow and added to tasks
        # if self._processing_wf is None:
        #     # Create a shared object if not provided
        #     if self._shared is None:
        #         self._shared = StreamShared()
        #
        #     # Create an OA workflow
        #     self._processing_wf = OAWorkflow(p_name='State Transition Wf', p_class_shared=self._shared)


        # 2. Create a reward task
        # if self._strans_task is None:
        #     # Create a pseudo reward task
        #     self._strans_task = OATask(p_name='Simulate Reaction',
        #         p_visualize=self._visualize,
        #         p_range_max=self.get_range(),
        #         p_duplicate_data=True)
        #
        #     # Assign the task method to custom implementation
        #     self._strans_task._run = self._run
        #
        #     # Add the task to workflow
        #     self._processing_wf.add_task(self._strans_task)


        # # 4. Creating task level attributes for states
        # try:
        #     self._strans_task._state
        # except AttributeError:
        #     self._strans_task._state = p_state.copy()
        #
        #
        # # 5. creating new instance with new state
        # self._instance = Instance(p_state)


        # 6. Run the workflow
        self._wf.run(p_inst_new=[p_state])


        # 7. Return the results
        return self._wf.get_so().get_results()[self.get_id()]


## -------------------------------------------------------------------------------------------------
    def _adapt(self, **p_kwargs) -> bool:
        """

        Parameters
        ----------
        p_kwargs

        Returns
        -------
        adapted: bool
            Returns true if the Function has adapted
        """

        adapted = False
        try:
            adapted = self._wf.adapt(**p_kwargs) or adapted
        except:
            adapted = adapted or False
        try:
            adapted = self._afct_strans.adapt(**p_kwargs) or adapted
        except:
            adapted = adapted or False
        return adapted

## -------------------------------------------------------------------------------------------------
    def _adapt_on_event(self, p_event_id:str, p_event_object:Event) -> bool:
        """

        Parameters
        ----------
        p_event_id
        p_event_object

        Returns
        -------

        """
        pass


## -------------------------------------------------------------------------------------------------
    def add_task(self, p_task:StreamTask, p_pred_task = None):
        """

        Parameters
        ----------
        p_task

        Returns
        -------

        """
        self._wf.add_task(p_task, p_pred_task)


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst_new, p_inst_del):
        """

        Parameters
        ----------
        p_inst_new
        p_inst_del

        Returns
        -------

        """
        try:
            self.get_so().add_result(self.get_id(), AFctSTrans.simulate_reaction(self,
                                                                                p_state=self._state,
                                                                                p_action=self._action))
        except:
            self.get_so().add_result(self.get_id(), FctSTrans.simulate_reaction(self,
                                                                            p_state=self._state,
                                                                            p_action=self._action))


## -------------------------------------------------------------------------------------------------
    def _setup_fct_workflow(self):
        """

        Returns
        -------

        """

        self._wf.add_task(PseudoTask(p_wrap_method = self._run))
        return False







## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAFctSuccess(FctSuccess, Model):
    """

    Parameters
    ----------
    p_name
    p_range_max
    p_class_shared
    p_visualize
    p_logging
    p_kwargs
    """


    ## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_afct_cls = None,
                 p_state_space: MSpace = None,
                 p_action_space: MSpace = None,
                 p_input_space_cls=ESpace,
                 p_output_space_cls=ESpace,
                 p_output_elem_cls=State,  # Specific output element type
                 p_threshold=0,
                 p_buffer_size=0,
                 p_ada:bool=True,
                 # p_visualize:bool=False,
                 # p_logging=Log.C_LOG_ALL,
                 # p_name:str=None,
                 # p_range_max=Async.C_RANGE_THREAD,
                 # p_class_shared=None,
                 p_wf_success: OAWorkflow = None,
                 p_visualize:bool=False,
                 p_logging=Log.C_LOG_ALL,
                 **p_kwargs):

        self._afct_success = None
        if p_afct_cls is not None:
            if (p_state_space is None) or (p_action_space is None):
                raise ParamError("Please provide mandatory parameters state and action space.")

            self._afct_success = AFctSuccess(p_afct_cls=p_afct_cls,
                                            p_state_space=p_state_space,
                                            p_action_space=p_action_space,
                                            p_input_space_cls=p_input_space_cls,
                                            p_output_space_cls=p_output_space_cls,
                                            p_output_elem_cls=p_output_elem_cls,
                                            p_threshold=p_threshold,
                                            p_buffer_size=p_buffer_size,
                                            p_ada=p_ada,
                                            p_visualize=p_visualize,
                                            p_logging=p_logging,
                                            **p_kwargs)
                #
        # else:
        FctSuccess.__init__(self, p_logging=p_logging)

        Model.__init__(self)
        self._wf_success = p_wf_success
        # self._shared = p_class_shared
        self._success_task = None
        self._instance:Instance = None
        self._state:State = None

## -------------------------------------------------------------------------------------------------
    def compute_success(self, p_state: State) -> bool:
        """

        Parameters
        ----------
        p_state

        Returns
        -------

        """
        self.log(Log.C_LOG_TYPE_I, 'Assessing Success...')

        # # 1. check if the user has already created a workflow and added to tasks
        # if self._processing_wf is None:
        #     # Create a shared object if not provided
        #     if self._shared is None:
        #         self._shared = StreamShared()
        #
        #     # Create an OA workflow
        #     self._processing_wf = OAWorkflow(p_name='Success Assessment Wf', p_class_shared=self._shared)
        #
        #
        # # 2. Create a reward task
        # if self._success_task is None:
        #     # Create a pseudo reward task
        #     self._success_task = OATask(p_name='Compute Success',
        #         p_visualize=self._visualize,
        #         p_range_max=self.get_range(),
        #         p_duplicate_data=True)
        #
        #     # Assign the task method to custom implementation
        #     self._success_task._run = self._run
        #
        #     # Add the task to workflow
        #     self._processing_wf.add_task(self._success_task)
        #
        #
        # # 4. Creating task level attributes for states
        # try:
        #     self._success_task._state
        # except AttributeError:
        #     self._success_task._state = p_state.copy()
        #
        #
        #
        # # 5. creating new instance with new state
        # self._instance = Instance(p_state)


        # 6. Run the workflow
        self._wf_success.run(p_inst_new=[p_state])


        # 7. Return the results
        return self._wf_success.get_so().get_results()[self.get_id()]


## -------------------------------------------------------------------------------------------------
    def _adapt_on_event(self, p_event_id:str, p_event_object:Event) -> bool:
        """

        Parameters
        ----------
        p_event_id
        p_event_object

        Returns
        -------

        """
        pass


## -------------------------------------------------------------------------------------------------
    def add_task(self, p_task:StreamTask, p_pred_tasks:list = None):
        """

        Parameters
        ----------
        p_task

        Returns
        -------

        """
        self._wf_success.add_task(p_task = p_task, p_pred_tasks=p_pred_tasks)


## -------------------------------------------------------------------------------------------------
    def _run_wf_success(self, p_inst_new, p_inst_del):
        """

        Parameters
        ----------
        p_inst_new
        p_inst_del

        Returns
        -------

        """
        try:
            self.get_so().add_result(self.get_id(), AFctSuccess.compute_success(self,
                                                                                 p_state=self._state))
        except:
            self.get_so().add_result(self.get_id(), FctSuccess.compute_success(self,
                                                                            p_state=self._state))


## -------------------------------------------------------------------------------------------------
    def _adapt(self, **p_kwargs) -> bool:
        """

        Parameters
        ----------
        p_kwargs

        Returns
        -------

        """
        adapted = False
        try:
            adapted = self._wf_success.adapt(**p_kwargs) or adapted
        except:
            adapted = False or adapted
        try:
            adapted = self._afct_success.adapt(**p_kwargs) or adapted
        except:
            adapted = False or adapted
        return adapted


## -------------------------------------------------------------------------------------------------
    def _setup_fct_workflow(self):
        """

        Returns
        -------

        """

        self._wf_success.add_task(PseudoTask(p_wrap_method = self._run_wf_success))
        return False







## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAFctBroken(FctBroken, Model):
    """

    Parameters
    ----------
    p_name
    p_range_max
    p_class_shared
    p_visualize
    p_logging
    p_kwargs
    """


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_afct_cls = None,
                 p_state_space: MSpace = None,
                 p_action_space: MSpace = None,
                 p_input_space_cls=ESpace,
                 p_output_space_cls=ESpace,
                 p_output_elem_cls=State,  # Specific output element type
                 p_threshold=0,
                 p_buffer_size=0,
                 p_ada:bool=True,
                 # p_visualize:bool=False,
                 # p_logging=Log.C_LOG_ALL,
                 # p_name:str=None,
                 # p_range_max=Async.C_RANGE_THREAD,
                 # p_class_shared=None,
                 p_wf_broken: OAWorkflow = None,
                 p_visualize:bool=False,
                 p_logging=Log.C_LOG_ALL,
                 **p_kwargs):

        self._afct_broken = None
        if p_afct_cls is not None:
            if (p_state_space is None) or (p_action_space is None):
                raise ParamError("Please provide mandatory parameters state and action space.")

            self._afct_broken = AFctSuccess(p_afct_cls=p_afct_cls,
                                            p_state_space=p_state_space,
                                            p_action_space=p_action_space,
                                            p_input_space_cls=p_input_space_cls,
                                            p_output_space_cls=p_output_space_cls,
                                            p_output_elem_cls=p_output_elem_cls,
                                            p_threshold=p_threshold,
                                            p_buffer_size=p_buffer_size,
                                            p_ada=p_ada,
                                            p_visualize=p_visualize,
                                            p_logging=p_logging,
                                            **p_kwargs)
        #
        # else:
        FctBroken.__init__(self, p_logging=p_logging)

        Model.__init__(self)
        # self._broken_task:StreamTask = None
        self._wf_broken = p_wf_broken
        # self._shared = p_class_shared
        self._instance:Instance = None
        self._state:State = None


## -------------------------------------------------------------------------------------------------
    def compute_broken(self, p_state: State) -> bool:
        """

        Parameters
        ----------
        p_state

        Returns
        -------

        """
        self.log(Log.C_LOG_TYPE_I, 'Assessing Broken...')


        # # 1. check if the user has already created a workflow and added to tasks
        # if self._processing_wf is None:
        #     # Create a shared object if not provided
        #     if self._shared is None:
        #         self._shared = StreamShared()
        #
        #
        #     # Create an OA workflow
        #     self._processing_wf = OAWorkflow(p_name='Broken Assessment Wf', p_class_shared=self._shared)
        #
        #
        # # 2. Create a reward task
        # if self._broken_task is None:
        #     # Create a pseudo reward task
        #     self._broken_task = OATask(p_name='Compute Broken',
        #         p_visualize=self._visualize,
        #         p_range_max=self.get_range(),
        #         p_duplicate_data=True)
        #
        #     # Assign the task method to custom implementation
        #     self._broken_task._run = self._run
        #
        #     # Add the task to workflow
        #     self._processing_wf.add_task(self._broken_task)
        #
        #
        # # 4. Creating task level attributes for states
        # try:
        #     self._broken_task._state
        # except AttributeError:
        #     self._broken_task._state = p_state.copy()
        #
        #
        # # 5. creating new instance with new state
        # self._instance = Instance(p_state)


        # 6. Run the workflow
        self._wf_broken.run(p_inst_new=[p_state])


        # 7. Return the results
        return self._wf_broken.get_so().get_results()[self.get_id()]


## -------------------------------------------------------------------------------------------------
    def add_task(self, p_task: StreamTask, p_pred_tasks:list = None):
        """

        Parameters
        ----------
        p_task

        Returns
        -------

        """
        self._wf_broken.add_task(p_task = p_task, p_pred_tasks = p_pred_tasks)


## -------------------------------------------------------------------------------------------------
    def _run_wf_broken(self, p_inst_new, p_inst_del):
        """

        Parameters
        ----------
        p_inst_new
        p_inst_del

        Returns
        -------

        """
        try:
            self.get_so().add_result(self.get_id(), AFctBroken.compute_broken(self,
                                                                         p_state=self._state))
        except:
            self.get_so().add_result(self.get_id(), FctBroken.compute_broken(self,
                                                                         p_state=self._state))


## -------------------------------------------------------------------------------------------------
    def _adapt(self, **p_kwargs) -> bool:
        """

        Parameters
        ----------
        p_kwargs

        Returns
        -------

        """
        adapted = False
        try:
            adapted = self._wf_broken.adapt(**p_kwargs) or adapted
        except:
            adapted = False or adapted
        try:
            adapted = self._afct_broken.adapt(self, **p_kwargs) or adapted
        except:
            adapted = False or adapted
        return adapted


## -------------------------------------------------------------------------------------------------
    def _adapt_on_event(self, p_event_id:str, p_event_object:Event) -> bool:
        """

        Parameters
        ----------
        p_event_id
        p_event_object

        Returns
        -------

        """
        pass


## -------------------------------------------------------------------------------------------------
    def _setup_fct_workflow(self):
        """

        Returns
        -------

        """

        self._wf_broken.add_task(PseudoTask(p_wrap_method = self._run_wf_broken))
        return False







## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OASystem(OAFctBroken, OAFctSTrans, OAFctSuccess, ASystem):
    """

    Parameters
    ----------
    p_mode
    p_latency
    p_fct_strans
    p_fct_success
    p_fct_broken
    p_processing_wf
    p_visualize
    p_logging
    """


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_mode = Mode.C_MODE_SIM,
                 p_latency = None,
                 p_fct_strans : FctSTrans = None,
                 p_fct_success : FctSuccess = None,
                 p_fct_broken : FctBroken = None,
                 p_wf : OAWorkflow = None,
                 p_wf_success : OAWorkflow = None,
                 p_wf_broken : OAWorkflow = None,
                 p_visualize : bool = False,
                 p_logging = Log.C_LOG_ALL):

        ASystem.__init__(self,
                         p_mode = p_mode,
                         p_latency = p_latency,
                         p_fct_strans = p_fct_strans,
                         p_fct_success = p_fct_success,
                         p_fct_broken = p_fct_broken,
                         p_visualize = p_visualize,
                         p_logging = p_logging)

        OAFctSTrans.__init__(self, p_wf=p_wf)

        OAFctSuccess.__init__(self, p_wf=p_wf_success)

        OAFctBroken.__init__(self, p_wf_broken=p_wf_broken)



## -------------------------------------------------------------------------------------------------
    @staticmethod
    def setup_spaces():

        return None, None


## -------------------------------------------------------------------------------------------------
    def _set_adapted(self, p_adapted:bool):
        """

        Parameters
        ----------
        p_adapted

        Returns
        -------

        """
        pass


## -------------------------------------------------------------------------------------------------
    def _adapt(self, **p_kwargs) -> bool:
        """

        Parameters
        ----------
        p_kwargs

        Returns
        -------

        """
        adapted = False

        try:
            adapted = self._wf.adapt(**p_kwargs)
        except:
            adapted = False or adapted
        try:
            adapted = self._wf_success.adapt(**p_kwargs)
        except:
            adapted = False or adapted
        try:
            adapted = self._wf_broken.adapt(**p_kwargs)
        except:
            adapted = False or adapted
        try:
            adapted = self._fct_strans.adapt(**p_kwargs)
        except:
            adapted = False or adapted
        try:
            adapted = self._fct_success.adapt(**p_kwargs)
        except:
            adapted = False or adapted
        try:
            adapted = self._fct_broken.adapt(**p_kwargs)
        except:
            adapted = False or adapted

        return adapted



## -------------------------------------------------------------------------------------------------
    def switch_adaptivity(self, p_ada:bool):
        """

        Parameters
        ----------
        p_ada

        Returns
        -------

        """
        pass


## -------------------------------------------------------------------------------------------------
    def _adapt_on_event(self, p_event_id:str, p_event_object:Event) -> bool:
        """

        Parameters
        ----------
        p_event_id
        p_event_object

        Returns
        -------

        """
        pass


## -------------------------------------------------------------------------------------------------
    def add_task(self, p_task:OATask, p_pred_tasks:list = None):

        OAFctSuccess.add_task(self, p_task=p_task, p_pred_tasks=p_pred_tasks)

## -------------------------------------------------------------------------------------------------
    def add_task_success(self, p_task:OATask, p_pred_tasks:list = None):

        OAFctSuccess.add_task(self, p_task=p_task, p_pred_tasks=p_pred_tasks)

## -------------------------------------------------------------------------------------------------
    def add_task_broken(self, p_task:OATask, p_pred_tasks:list = None):

        OAFctBroken.add_task(self, p_task=p_task, p_pred_tasks=p_pred_tasks)

## -------------------------------------------------------------------------------------------------
    def simulate_reaction(self, p_state: State, p_action: Action, p_t_step : timedelta = None) -> State:

        if self._fct_strans is not None:
            return self._fct_strans.simulate_reaction(p_state, p_action, p_t_step)
        else:
            return OAFctSTrans.simulate_reaction(self, p_state, p_action, p_t_step)

## -------------------------------------------------------------------------------------------------
    def compute_success(self, p_state: State) -> bool:

        if self._fct_success is not None:
            return self._fct_success.compute_success(p_state)
        else:
            return OAFctSuccess.compute_success(self, p_state)

## -------------------------------------------------------------------------------------------------
    def compute_broken(self, p_state: State) -> bool:

        if self._fct_broken is not None:
            return self._fct_broken.compute_broken(p_state)
        else:
            return OAFctBroken.compute_broken(self, p_state)