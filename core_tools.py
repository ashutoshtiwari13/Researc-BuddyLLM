from utils.functional_utils import HotReload

def get_core_tools():
    from core_functions.analysis_jupyter_notebook import parsingIpynbFiles

    function_plugins = {
         "Parsing Jupyter Notebook Files": {
            "Color": "stop",
            "AsButton":False,
            "Function": HotReload(parsingIpynbFiles),
            "AdvancedArgs": True, # When calling, invoke the advanced parameter input area (default False)
            "ArgsReminder": "If you enter 0, the Markdown block in the notebook will not be parsed", # The display prompt of the advanced parameter input area
        },
    }


    