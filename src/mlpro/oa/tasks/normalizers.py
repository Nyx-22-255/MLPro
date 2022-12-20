## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.oa.tasks.normalizers
## -- Module  : normalizers.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-12-07  1.0.0     LSB      Creation/Release
## -- 2022-12-13  1.0.1     LSB      Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2022-12-07)
This module provides implementation for adaptive normalizers for MinMax Normalization and ZTransformation
"""


from mlpro.oa.models import *
from mlpro.bf.math import normalizers as Norm



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class NormalizerMinMax(OATask, Norm.NormalizerMinMax):
    """
    Class with functionality for adaptive normalization of instances using MinMax Normalization.

    Parameters
    ----------
    p_name: str, optional
        Name of the task.
    p_range_max:
        Processing range of the task, default is a Thread.
    p_ada:
        True if the task has adaptivity, default is true.
    p_visualize:
        True for visualization, false by default.
    p_logging:
        Logging level of the task. Default is Log.C_LOG_ALL
    p_kwargs:
        Additional task parameters
    """


## -------------------------------------------------------------------------------------------------
    def __init__(self,p_name: str = None,
                  p_range_max = StreamTask.C_RANGE_THREAD,
                  p_ada : bool = True,
                  p_visualize:bool = False,
                  p_logging = Log.C_LOG_ALL,
                  **p_kwargs):

        OATask.__init__(self,
                        p_name = p_name,
                        p_range_max = p_range_max,
                        p_ada = p_ada,
                        p_visualize = p_visualize,
                        p_logging=p_logging,
                        **p_kwargs)

        Norm.NormalizerMinMax.__init__(self)



## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst_new:list, p_inst_del:list):
        """
        Custom method to for run MinMax Normalizer task for normalizing new instances and denormalizing deleted
        instances.

        Parameters
        ----------
        p_inst_new: list
            List of new instances in the workflow
        p_inst_del: list
            List of deleted instances in the workflow

        """
        for i,inst in enumerate(p_inst_new):
            normalized_element = self.normalize(inst.get_feature_data())
            inst.get_feature_data().set_values(normalized_element)

        for j, del_inst in enumerate(p_inst_del):
            normalized_element = self.normalize(del_inst.get_feature_data())
            del_inst.get_feature_data().set_values(normalized_element)



## -------------------------------------------------------------------------------------------------
    def _adapt_on_event(self, p_event_id:str, p_event_object:Event) -> bool:
        """
        Custom method to adapt the MinMax normalizer parameters based on event raised by Boundary object for changed
        boundaries.

        Parameters
        ----------
        p_event_id: str
            Event id of the raised event

        p_event_obj: Event
            Event object that raises the corresponding event

        Returns
        -------
        adapted: bool
            Returns True, if the task has adapted. False otherwise.
        """


        inst_new = p_event_object.get_data()['p_inst_new']
        for i in inst_new:
            set = i.get_feature_data().get_related_set()
            break
        self.update_parameters(set)


        return True





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class NormalizerZTransform(OATask, Norm.NormalizerZTrans):
    """
    Class with functionality of adaptive normalization of instances with Z-Transformation

    Parameters
    ----------
    p_name: str, optional
        Name of the task.
    p_range_max:
        Processing range of the task, default is a Thread.
    p_ada:
        True if the task has adaptivity, default is true.
    p_visualize:
        True for visualization, false by default.
    p_logging:
        Logging level of the task. Default is Log.C_LOG_ALL
    p_kwargs:
        Additional task parameters
    """


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_name: str = None,
                 p_range_max=StreamTask.C_RANGE_THREAD,
                 p_ada: bool = True,
                 p_logging=Log.C_LOG_ALL,
                 **p_kwargs):

        OATask.__init__(self,
            p_name=p_name,
            p_range_max=p_range_max,
            p_ada=p_ada,
            p_logging=p_logging,
            **p_kwargs)

        Norm.NormalizerZTrans.__init__(self)


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst_new:list, p_inst_del:list):
        """
        Custom method to for run Z-transform task for normalizing new instances and denormalizing deleted instances.

        Parameters
        ----------
        p_inst_new: list
            List of new instances in the workflow
        p_inst_del: list
            List of deleted instances in the workflow

        """
        self.adapt(p_inst_new=p_inst_new, p_inst_del=p_inst_del)

        for i, inst in enumerate(p_inst_new):
            normalized_element = self.normalize(inst.get_feature_data())
            inst.get_feature_data().set_values(normalized_element)

        for i,del_inst in enumerate(p_inst_del):
            normalized_element = self.normalize(del_inst.get_feature_data())
            del_inst.get_feature_data().set_values(normalized_element)


## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_inst_new:list, p_inst_del:list) -> bool:
        """
        Custom method to for adapting of Z-transform parameters on new and deleted instances.

        Parameters
        ----------
        p_inst_new: list
            List of new instances in the workflow
        p_inst_del: list
            List of deleted instances in the workflow

        Returns
        -------
        adapted : bool
            Returns True, if task has adapted.

        """

        adapted = False
        try:
            # 1. Update parameters based on new elements
            for inst in p_inst_new:
                self.update_parameters(p_data_new=inst.get_feature_data())

            # 2. Update parameters based on deleted elements
            for del_inst in p_inst_del:
                self.update_parameters(p_data_del=del_inst.get_feature_data())

            adapted = True

        except: pass

        return adapted
