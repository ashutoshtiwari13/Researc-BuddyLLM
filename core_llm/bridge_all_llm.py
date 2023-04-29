"""
This file mainly contains 2 functions, which are the common interface of all LLMs. 
They will continue to call the lower-level LLM model to handle details such as multi-model parallelism.

     Functions without multi-threading capability: used in normal dialogues, have complete interactive functions, and cannot be multi-threaded
     1. predict(...)

     Functions capable of multi-thread calling: called in function plug-ins, flexible and concise
     2. predict_no_ui_long_connection(...)
"""

import tiktoken
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from utils.functional_util import get_conf,trimmed_format_exc

from .chatGPT_llm import predict_no_ui_long_connection as chatgpt_noui
from .chatGPT_llm import predict as chatgpt_ui

colors = ['#FF00FF', '#00FFFF', '#FF0000', '#990099', '#009999', '#990044']

class LazyLoadTiktoken(object):
     def __init__(self,model):
          self.model = model

     def get_encoder(model):
          print('The tokenizer is being loaded, if it is the first run, it may take a while to download the parameters')
          tmp = tiktoken.encoding_for_model(model)
          prrint('The tokenizer is loaded')
          return tmp

     def encode(self,*args,**kwargs):
          encoder = self.get_encoder(self.model)
          return encoder.encode(*args,**kwargs)

     def decode():
          encoder = self.get_encoder(self.model)
          return encoder.encode(*args,**kwargs)


#Model Endpoint
API_URL_REDIRECT, = get_conf("API_URL_REDIRECT")
openai_endpoint = "https://api.openai.com/v1/chat/completions"

try:
    API_URL, = get_conf("API_URL")
    if API_URL != "https://api.openai.com/v1/chat/completions": 
        openai_endpoint = API_URL
        print("warn! The API_URL configuration option will be deprecated, please replace it with the API_URL_REDIRECT configuration")
except:
    pass

if openai_endpoint in API_URL_REDIRECT: openai_endpoint = API_URL_REDIRECT[openai_endpoint]

# get tokenizer
tokenizer_gpt35 = LazyloadTiktoken("gpt-3.5-turbo")
tokenizer_gpt4 = LazyloadTiktoken("gpt-4")
get_token_num_gpt35 = lambda txt: len(tokenizer_gpt35.encode(txt, disallowed_special=()))
get_token_num_gpt4 = lambda txt: len(tokenizer_gpt4.encode(txt, disallowed_special=()))

model_info ={
     #gptOpenAi
     "gpt-3.5-turbo": {
        "fn_with_ui": chatgpt_ui,
        "fn_without_ui": chatgpt_noui,
        "endpoint": openai_endpoint,
        "max_token": 4096,
        "tokenizer": tokenizer_gpt35,
        "token_cnt": get_token_num_gpt35,
    },

    "gpt-4": {
        "fn_with_ui": chatgpt_ui,
        "fn_without_ui": chatgpt_noui,
        "endpoint": openai_endpoint,
        "max_token": 8192,
        "tokenizer": tokenizer_gpt4,
        "token_cnt": get_token_num_gpt4,
    },

}

def LLM_CATCH_EXCEPTION(func):
    """
    Decorator function to display errors
    """
    def reDesign(inputs, llm_kwargs, history, sys_prompt, observe_window, console_slience):
        try:
            return func(inputs, llm_kwargs, history, sys_prompt, observe_window, console_slience)
        except Exception as e:
            tb_str = '\n```\n' + trimmed_format_exc() + '\n```\n'
            observe_window[0] = tb_str
            return tb_str
    return reDesign



def predict_no_ui_long_connection(inputs, llm_kwargs, history, sys_prompt, observe_window, console_slience=False):
    """
    Send to LLM, wait for reply, complete in one go, no intermediate process displayed. But the method of stream is 
    used internally to avoid the network cable being pinched in the middle.  

     inputs:
         is the input for this query
     sys_prompt:
         System silent prompt
     llm_kwargs:
         Internal tuning parameters of LLM
     history:
         is a list of previous conversations
     observe_window = None:
         It is responsible for passing the output part across threads. Most of the time, it is only for the fancy visual effect, and it can be left blank.
          observe_window[0]: observation window. observe_window[1]: watchdog
    """
    import threading, time, copy

    model = llm_kwargs['llm_model']
    n_model = 1
    if '&' not in model:
        assert not model.startswith("tgui"), "TGUI does not support the implementation of function plugins"

        # If only 1 big language model is asked:
        method = model_info[model]["fn_without_ui"]
        return method(inputs, llm_kwargs, history, sys_prompt, observe_window, console_slience)
    else:
        # If asking multiple large language models simultaneously:
        executor = ThreadPoolExecutor(max_workers=4)
        models = model.split('&')
        n_model = len(models)
        
        window_len = len(observe_window)
        assert window_len==3
        window_mutex = [["", time.time(), ""] for _ in range(n_model)] + [True]

        futures = []
        for i in range(n_model):
            model = models[i]
            method = model_info[model]["fn_without_ui"]
            llm_kwargs_feedin = copy.deepcopy(llm_kwargs)
            llm_kwargs_feedin['llm_model'] = model
            future = executor.submit(LLM_CATCH_EXCEPTION(method), inputs, llm_kwargs_feedin, history, sys_prompt, window_mutex[i], console_slience)
            futures.append(future)

        def mutex_manager(window_mutex, observe_window):
            while True:
                time.sleep(0.25)
                if not window_mutex[-1]: break
                # watchdog
                for i in range(n_model): 
                    window_mutex[i][1] = observe_window[1]
                # Observation window (window)
                chat_string = []
                for i in range(n_model):
                    chat_string.append( f"【{str(models[i])} explain】: <font color=\"{colors[i]}\"> {window_mutex[i][0]} </font>" )
                res = '<br/><br/>\n\n---\n\n'.join(chat_string)
                # # # # # # # # # # #
                observe_window[0] = res

        t_model = threading.Thread(target=mutex_manager, args=(window_mutex, observe_window), daemon=True)
        t_model.start()

        return_string_collect = []
        while True:
            worker_done = [h.done() for h in futures]
            if all(worker_done):
                executor.shutdown()
                break
            time.sleep(1)

        for i, future in enumerate(futures):  # wait and get
            return_string_collect.append( f"【{str(models[i])} Explain: <font color=\"{colors[i]}\"> {future.result()} </font>" )

        window_mutex[-1] = False # stop mutex thread
        res = '<br/><br/>\n\n---\n\n'.join(return_string_collect)
        return res


def predict(inputs, llm_kwargs, *args, **kwargs):
    """
     Send to LLM to stream the output.
     Used for basic dialog functionality.
     inputs is the input of this query
     top_p, temperature are the internal tuning parameters of LLM
     history is a list of previous conversations (note that whether it is inputs or history, if the content is too long, it will trigger an error that the number of tokens overflows)
     chatbot is the dialog list displayed in the WebUI, modify it, and then yeild out, you can directly modify the content of the dialog interface
     additional_fn represents which button is clicked, see functional.py for the button
    """

    method = model_info[llm_kwargs['llm_model']]["fn_with_ui"]
    yield from method(inputs, llm_kwargs, *args, **kwargs)
