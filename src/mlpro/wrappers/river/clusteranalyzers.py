## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.wrappers.river
## -- Module  : clusteranalyzers.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-05-12  0.0.0     DA       Creation
## -- 2023-05-xx  1.0.0     SY       First version release
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-05-xx)

This module provides wrapper classes from River to MLPro, specifically for cluster analyzers. This
module includes three clustering algorithms from River that are embedded to MLPro, such as:

1) DBSTREAM (https://riverml.xyz/latest/api/cluster/DBSTREAM/)

2) CluStream (https://riverml.xyz/latest/api/cluster/CluStream/)

3) DenStream (https://riverml.xyz/latest/api/cluster/DenStream/)

Learn more:
https://www.riverml.xyz/

"""


from mlpro.wrappers.river.basics import WrapperRiver
from mlpro.oa.streams.tasks.clusteranalyzers import ClusterAnalyzer, Cluster, ClusterCentroid
from mlpro.bf.mt import Task as MLTask
from mlpro.bf.various import Log
from mlpro.bf.streams import *
from river import base, cluster
from typing import List, Tuple





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrClusterAnalyzerRiver2MLPro (WrapperRiver, ClusterAnalyzer):

    C_TYPE              = 'River Cluster Analyzer'
    C_NAME              = '????'
    
    C_WRAPPED_PACKAGE   = 'river'
    C_MINIMUM_VERSION   = '0.15.0'
    
    C_CLS_CLUSTER       = Cluster


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_river_algo:base.Clusterer,
                 p_name:str = None,
                 p_range_max = MLTask.C_RANGE_THREAD,
                 p_ada:bool = True,
                 p_visualize:bool = False,
                 p_logging = Log.C_LOG_ALL,
                 **p_kwargs):
        
        self._river_algo = p_river_algo

        WrapperRiver.__init__(self, p_logging=p_logging)

        ClusterAnalyzer.__init__(self,
                                 p_name=p_name,
                                 p_range_max=p_range_max,
                                 p_ada=p_ada,
                                 p_visualize=p_visualize,
                                 **p_kwargs)


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst_new: List[Instance], p_inst_del: List[Instance]):
        # p_inst_del has no use

        # transform new instance to a dictionary of features
        x = {0: 1, 1: 1} # example

        self._river_algo.learn_one(x)


## -------------------------------------------------------------------------------------------------
    def get_clusters(self):
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def get_cluster_membership(self, p_inst:Instance) -> List[Tuple[str, float, Cluster]]:
        # to be added
        pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrRiverDBStream2MLPro (WrClusterAnalyzerRiver2MLPro):

    C_NAME              = 'DBSTREAM'
    
    C_CLS_CLUSTER       = ClusterCentroid


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_name:str = None,
                 p_range_max = MLTask.C_RANGE_THREAD,
                 p_ada:bool = True,
                 p_visualize:bool = False,
                 p_logging = Log.C_LOG_ALL,
                 p_clustering_threshold:float = 1.0,
                 p_fading_factor:float = 0.01,
                 p_cleanup_interval:float = 2,
                 p_intersection_factor:float = 0.3,
                 p_minimum_weight:float = 1.0,
                 **p_kwargs):
        
        alg = cluster.DBSTREAM(clustering_threshold=p_clustering_threshold,
                               fading_factor=p_fading_factor,
                               cleanup_interval=p_cleanup_interval,
                               intersection_factor=p_intersection_factor,
                               minimum_weight=p_minimum_weight)

        super().__init__(p_river_algo=alg,
                         p_name=p_name,
                         p_range_max=p_range_max,
                         p_ada=p_ada,
                         p_visualize=p_visualize,
                         p_logging=p_logging,
                         **p_kwargs)


## -------------------------------------------------------------------------------------------------
    def get_clusters(self):
        # to be added
        pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrRiverCluStream2MLPro (WrClusterAnalyzerRiver2MLPro):

    C_NAME              = 'CluStream'
    
    C_CLS_CLUSTER       = ClusterCentroid


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_name:str = None,
                 p_range_max = MLTask.C_RANGE_THREAD,
                 p_ada:bool = True,
                 p_visualize:bool = False,
                 p_logging = Log.C_LOG_ALL,
                 p_n_macro_clusters:int = 5,
                 p_max_micro_clusters:int = 100,
                 p_micro_cluster_r_factor:int = 2,
                 p_time_window:int = 1000,
                 p_time_gap:int = 100,
                 p_seed:int = None,
                 p_halflife:float = 0.5,
                 p_mu:float = 1,
                 p_sigma:float = 1,
                 p_p:int = 2,
                 **p_kwargs):
        
        alg = cluster.CluStream(n_macro_clusters=p_n_macro_clusters,
                                max_micro_clusters=p_max_micro_clusters,
                                micro_cluster_r_factor=p_micro_cluster_r_factor,
                                time_window=p_time_window,
                                time_gap=p_time_gap,
                                seed=p_seed,
                                halflife=p_halflife,
                                mu=p_mu,
                                sigma=p_sigma,
                                p=p_p)

        super().__init__(p_river_algo=alg,
                         p_name=p_name,
                         p_range_max=p_range_max,
                         p_ada=p_ada,
                         p_visualize=p_visualize,
                         p_logging=p_logging,
                         **p_kwargs)


## -------------------------------------------------------------------------------------------------
    def get_clusters(self):
        # to be added
        pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrRiverDenStream2MLPro (WrClusterAnalyzerRiver2MLPro):

    C_NAME              = 'DenStream'
    
    C_CLS_CLUSTER       = ClusterCentroid


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_name:str = None,
                 p_range_max = MLTask.C_RANGE_THREAD,
                 p_ada:bool = True,
                 p_visualize:bool = False,
                 p_logging = Log.C_LOG_ALL,
                 p_decaying_factor:float = 0.25,
                 p_beta:float = 0.75,
                 p_mu:float = 2,
                 p_epsilon:float = 0.02,
                 p_n_samples_init:int = 1000,
                 p_stream_speed:int = 100,
                 **p_kwargs):
        
        alg = cluster.DenStream(decaying_factor=p_decaying_factor,
                                beta=p_beta,
                                mu=p_mu,
                                epsilon=p_epsilon,
                                n_samples_init=p_n_samples_init,
                                stream_speed=p_stream_speed)

        super().__init__(p_river_algo=alg,
                         p_name=p_name,
                         p_range_max=p_range_max,
                         p_ada=p_ada,
                         p_visualize=p_visualize,
                         p_logging=p_logging,
                         **p_kwargs)


## -------------------------------------------------------------------------------------------------
    def get_clusters(self):
        # to be added
        pass


    