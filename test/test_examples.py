## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro
## -- Module  : test_example
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-10-06  1.0.0     MRD      Creation
## -- 2021-10-06  1.0.0     MRD      Release First Version
## -- 2021-12-12  1.0.1     DA       Howto 17 added
## -- 2021-12-20  1.0.2     DA       Howto 08 disabled
## -- 2022-02-28  1.0.3     SY       Howto 06, 07 of basic functions are added
## -- 2022-02-28  1.0.4     SY       Howto 07 of basic functions is disabled
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.4 (2022-02-28)

Unit test for all examples available.
"""

import pytest
import importlib


howto_list = {
    "bf_001": "mlpro.bf.examples.howto_bf_001_logging",
    "bf_002": "mlpro.bf.examples.howto_bf_002_timer",
    "bf_003": "mlpro.bf.examples.howto_bf_003_spaces_and_elements",
    "bf_004": "mlpro.bf.examples.howto_bf_004_store_plot_and_save_variables",
    "bf_005": "mlpro.bf.examples.howto_bf_005_hyperparameters",
    "bf_006": "mlpro.bf.examples.howto_bf_006_buffers",
    "bf_007": "mlpro.bf.examples.howto_bf_007_hyperparameter_tuning_using_hyperopt",
    "bf_008": "mlpro.bf.examples.howto_bf_008_hyperparameter_tuning_using_optuna",
    "bf_009": "mlpro.bf.examples.howto_bf_009_sciui_reuse_of_interactive_2d_3d_input_space",
    "bf_010": "mlpro.bf.examples.howto_bf_010_sciui_reinforcement_learning_cockpit"
}


# import mlpro.bf.examples
# from pkgutil import iter_modules

# def list_submodules(module):
#     for submodule in iter_modules(module.__path__):
#         print(submodule.name)

# list_submodules(mlpro.bf.examples)


@pytest.mark.parametrize("cls", list(howto_list.keys()))
def test_howto(cls):
    importlib.import_module(howto_list[cls])

