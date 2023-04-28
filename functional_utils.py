import os, traceback
from functools import wraps,lru_cache

class ChatBotWithCookies(list):
    """
    This Class helps in implementing "memory" functionality of Research Buddy(TBD)
    """
    def __init__(self, cookies):
        self._cookies = cookie

    def write_list(self,list):
        for l in list:
            self.append(l)

    def get_list(self):
        return [l for l in self]

    def get_cookies(self):
        return self._cookies

def GenericArgsWrapper(func):
    """
    Recognize input parameters, Change order and the structure of the input params
    """
    def reDesign( cookies,max_length, llm_model, txt, txt2,top_p, temparature,chatbot, history, system_prompt, plugin_advanced_arg, *args):
        local_text = txt
        if txt == "" and txt2 != "": local_text = txt2
        #update cookies step
        cookies.update({
            'top_p': top_p,
            'temperature':temparature
        })

        llm_kwargs = {
            'api_key': cookies['api_key']
            'llm_model' : llm_model,
            'max_length': max_length,
            'top_p': top_p,
            'temprature': temparature
        }

        plugin_kwargs = {
            'plugin_advanced_arg':plugin_advanced_arg,
        }

        use_cookies = ChatBotWithCookies(cookies)
        use_cookies.write_list(chatbot)
        yield from func(local_text,llm_kwargs, plugin_kwargs, chatbot_with_cookie, history, system_prompt, *args)

    return reDesign    


def updateUI(chatbot, history, msg='normal', **kwargs):
    """
    Check and Updates the UI
    """
    assert isinstance(chatbot,ChatBotWithCookies)
    yield chatbot.get_cookies(), chatbot, history, msg



def trimmed_format_exc():
    str = traceback.format_exc()
    current_path = os.getcwd()
    replace_path = "."
    return str.replace(current_path, replace_path)

def CatchException(func):

    """
    Captures exception in function , encapsulates and displays in chat
    """
    @wraps(func)
    def reDesign(txt,top_p, temparature,chatbot, history, systemPromptTxt, WEB_PORT):
        try:
            yield from func(txt,top_p, temparature,chatbot, history, systemPromptTxt, WEB_PORT)
        except Exception as e:
            #check proxy connection and redirect
            from proxy_checker import proxy_checker
            from utils import get_conf
            proxies, = get_conf('proxies')
            tb_str = '```\n' + trimmed_format_exc() + '```'
            if chatbot is None or len(chatbot) == 0:
                chatbot = [["Proxy Scheduling exception", "Abnormal Temination"]]
            chatbot[-1] = (chatbot[-1][0],
                     f"[Local Message] Error: \n\n{tb_str} \n\n Agent Error: \n\n{check_proxy(proxies)}")   
            yield from updateUI(chatbot=chatbot, history=history, msg=f'Abnormal {e}')

    return reDesign

def HotReload(func):
    """
    To facilitate the real-time update of python function plugins 
    getattr() gets the funcction and Reloads the function in a new module
    yield from statement is used to return the reloaded function and execute on the reDesign function.
    Ultimately, the reDesign function returns the inner function. This internal function can update the 
    original definition of the function to the latest version and execute the new version of the function.
    Read about @wraps() from functools
    """
    @wraps(func):
    def reDesign(*args, **kwargs):
        func_name = func.__name___
        func_hot_reload = getattr(importlib.reload(inspect.getmodule(func)),func_name)
        yield from func_hot_reload(*args,**kwargs)
    return reDesign    










