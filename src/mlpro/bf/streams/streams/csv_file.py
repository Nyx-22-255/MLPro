## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.streams.streams
## -- Module  : csv_file.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-03-02  0.0.0     SY       Creation 
## -- 2023-03-06  1.0.0     SY       First release
## -- 2023-04-10  1.0.1     SY       Refactoring
## -- 2023-04-14  1.1.0     SY       Make StreamMLProCSV independent from StreamMLProBase
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2023-04-14)

This module provides the native stream class StreamMLProCSV.
This stream provides a functionality to convert csv file to a MLPro compatible stream data.
"""


import numpy as np
from mlpro.bf.data import *
from mlpro.bf.streams.models import *





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StreamMLProCSV(Stream):

    C_ID        = 'CSV2MLPro'
    C_NAME      = 'CSV Format to MLPro Stream'
    C_VERSION   = '1.0.0'

    C_SCIREF_TYPE   = ScientificObject.C_SCIREF_TYPE_ONLINE
    C_SCIREF_AUTHOR = 'MLPro'
    C_SCIREF_URL    = 'https://mlpro.readthedocs.io'


## -------------------------------------------------------------------------------------------------
    def set_options(self, **p_kwargs):
        """
        Method to set specific options for the stream. The possible options depend on the 
        stream provider and stream itself.
        """

        self._kwargs = p_kwargs.copy()

        if 'p_path_load' not in self._kwargs:
            self._kwargs['p_path_load'] = None
            
        if 'p_csv_filename' not in self._kwargs:
            self._kwargs['p_csv_filename'] = None
            
        if 'p_delimiter' not in self._kwargs:
            self._kwargs['p_delimiter'] = "\t"
            
        if 'p_frame' not in self._kwargs:
            self._kwargs['p_frame'] = True
            
        if 'p_header' not in self._kwargs:
            self._kwargs['p_header'] = True
            
        if 'p_list_features' not in self._kwargs:
            self._list_features = None
        else:
            self._list_features = self._kwargs['p_list_features']
            
        if 'p_list_labels' not in self._kwargs:
            self._list_labels = None
        else:
            self._list_labels = self._kwargs['p_list_labels']
    

## -------------------------------------------------------------------------------------------------
    def _setup_feature_space(self) -> MSpace:
        
        feature_space : MSpace = MSpace()
        
        if self._list_features is not None:
            for ftrs in self._list_features:
                if ftrs in self._from_csv.names:
                    feature_space.add_dim(Feature(p_name_short = ftrs,
                                                  p_base_set = Feature.C_BASE_SET_R,
                                                  p_name_long = ftrs,
                                                  p_name_latex = '',
                                                  p_description = '',
                                                  p_symmetrical = False,
                                                  p_logging=Log.C_LOG_NOTHING)
                                          )            
        return feature_space


## -------------------------------------------------------------------------------------------------
    def _setup_label_space(self) -> MSpace:
        
        label_space : MSpace = MSpace()
        
        if self._list_labels is not None:
            for ftrs in self._list_labels:
                if ftrs in self._from_csv.names:
                    label_space.add_dim(Label(p_name_short = ftrs,
                                              p_base_set = Feature.C_BASE_SET_R,
                                              p_name_long = ftrs,
                                              p_name_latex = '',
                                              p_description = '',
                                              p_symmetrical = False,
                                              p_logging=Log.C_LOG_NOTHING)
                                        )
        return label_space


## -------------------------------------------------------------------------------------------------
    def _init_dataset(self):
        
        p_variable      = []
        self._from_csv  = DataStoring(p_variable)
        self._from_csv.load_data(self._kwargs['p_path_load'],
                                 self._kwargs['p_csv_filename'],
                                 self._kwargs['p_delimiter'],
                                 self._kwargs['p_frame'],
                                 self._kwargs['p_header'])
        
        try:
            extended_data   = []
            key_0           = list(self._from_csv.memory_dict.keys())[0]
            for fr in self._from_csv.memory_dict[key_0]:
                extended_data.extend(self._from_csv.memory_dict[key_0][fr])
            self.C_NUM_INSTANCES = self._num_instances = len(extended_data)
        except:
            self.C_NUM_INSTANCES = self._num_instances = 0
        
        if self._sampler is not None:
            self._sampler.set_num_instances(self._num_instances)

        self._feature_space = self._setup_feature_space()
        self._label_space   = self._setup_label_space()
        
        dim             = self._feature_space.get_num_dim()
        dim_l           = self._label_space.get_num_dim()
        self._dataset   = np.zeros((self.C_NUM_INSTANCES,dim))
        self._dataset_l = np.zeros((self.C_NUM_INSTANCES,dim_l))
        extended_data   = {}
        ids             = self._feature_space.get_dim_ids()
        
        x = 0
        for id_ in ids:
            ft_name = self._feature_space.get_dim(id_).get_name_short()
            extended_data[ft_name] = []
            for fr in self._from_csv.memory_dict[ft_name]:
                extended_data[ft_name].extend(self._from_csv.memory_dict[ft_name][fr])
            self._dataset[:,x] = np.array(extended_data[ft_name])
            x += 1
        
        x = 0        
        ids = self._label_space.get_dim_ids()    
        for id_ in ids:
            lbl_name = self._label_space.get_dim(id_).get_name_short()
            extended_data[lbl_name] = []
            for fr in self._from_csv.memory_dict[lbl_name]:
                extended_data[lbl_name].extend(self._from_csv.memory_dict[lbl_name][fr])
            self._dataset_l[:,x] = np.array(extended_data[lbl_name])
            x += 1


## -------------------------------------------------------------------------------------------------
    def _reset(self):
        self._index = 0
        self._init_dataset()


## -------------------------------------------------------------------------------------------------
    def _get_next(self) -> Instance:

        if self._index == self.C_NUM_INSTANCES: raise StopIteration

        feature_data = Element(self._feature_space)
        feature_data.set_values(p_values=self._dataset[self._index])

        self._index += 1

        return Instance( p_feature_data=feature_data )