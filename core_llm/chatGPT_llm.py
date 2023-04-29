"""
This file mainly contains three functions

     Functions not capable of multithreading:
     1. predict: used in normal dialogue, with complete interactive functions, not multi-threaded

     Functions capable of multithreading
     2. predict_no_ui: advanced experimental function module call, will not be displayed on the interface in real time, the parameters are simple, can be multi-threaded in parallel, and it is convenient to realize complex function logic
     3. predict_no_ui_long_connection: During the experiment, it is found that when calling predict_no_ui to process long documents, the connection with openai is easy to break. This function uses stream to solve this problem and also supports multi-threading.
"""

import json
import time
import gradio as gr
import logging
import traceback
import requests
import importlib

# config_private.py put your own secrets such as API and proxy URL
# When reading, first check whether there is a private config_private configuration file (not controlled by git), if so, overwrite the original config file

from utils.functional_utils import get_conf, updateUI,trimmed_format_exc
from utils.keys.utils import is_any_api_key, select_api_key, what_keys
from utils.process_utils import clip_history
proxies, API_KEY, TIMEOUT_SECONDS, MAX_RETRY = \
    get_conf('proxies', 'API_KEY', 'TIMEOUT_SECONDS', 'MAX_RETRY')

timeout_bot_msg = '[Local Message] Request timeout. Network error. Please check proxy settings in config.py.' + \
                  'Network error, check whether the proxy server is available, and whether the format of the proxy setting is correct, the format must be [protocol]://[address]:[port], both are indispensable.'

def get_full_error(chunk, stream_response):
    """
        Get the complete error report returned from Openai
    """
    while True:
        try:
            chunk += next(stream_response)
        except:
            break
    return chunk


def generate_payload(inputs, llm_kwargs, history, system_prompt, stream):
    """
    Integrate all information, select LLM model, generate http request, prepare for sending request
    """
    if not is_any_api_key(llm_kwargs['api_key']):
        raise AssertionError("You provided wrong API_KEY. \n\n1. Temporary solution: Type api_key directly in the input area, then press Enter to submit. \n\n2. Long-term solution: configure in config.py.")

    api_key = select_api_key(llm_kwargs['api_key'],llm_kwargs['llm_model'])

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    conversation_cnt = len(history) // 2

    messages = [{"role": "system", "content": system_prompt}]
    if conversation_cnt:
        for index in range(0, 2*conversation_cnt, 2):
            what_i_have_asked = {}
            what_i_have_asked["role"] = "user"
            what_i_have_asked["content"] = history[index]
            what_gpt_answer = {}
            what_gpt_answer["role"] = "assistant"
            what_gpt_answer["content"] = history[index+1]
            if what_i_have_asked["content"] != "":
                if what_gpt_answer["content"] == "": continue
                if what_gpt_answer["content"] == timeout_bot_msg: continue
                messages.append(what_i_have_asked)
                messages.append(what_gpt_answer)
            else:
                messages[-1]['content'] = what_gpt_answer['content']


    what_i_ask_now = {}
    what_i_ask_now["role"] = "user"
    what_i_ask_now["content"] = inputs
    messages.append(what_i_ask_now)

    payload = {
        "model": llm_kwargs['llm_model'].strip('api2d-'),
        "messages": messages, 
        "temperature": llm_kwargs['temperature'],  # 1.0,
        "top_p": llm_kwargs['top_p'],  # 1.0,
        "n": 1,
        "stream": stream,
        "presence_penalty": 0,
        "frequency_penalty": 0,
    }

    try:
        print(f" {llm_kwargs['llm_model']} : {conversation_cnt} : {inputs[:100]} ..........")
    except:
        print('There may be garbled characters in the input.')
    return headers,payload    


def predict(inputs, llm_kwargs, plugin_kwargs, chatbot, history=[], system_prompt='', stream = True, additional_fn=None):
    """
     Send to chatGPT to stream the output.
     Used for basic dialog functionality.
     inputs is the input of this query
     top_p, temperature are internal tuning parameters of chatGPT
     history is a list of previous conversations (note that whether it is inputs or history, if the content is too long, it will trigger an error that the number of tokens overflows)
     chatbot is the dialog list displayed in the WebUI, modify it, and then yeild out, you can directly modify the content of the dialog interface
     additional_fn represents which button is clicked, see functional.py for the button
    """
    if is_any_api_key(inputs):
        chatbot._cookies['api_key'] = inputs
        chatbot.append(("Enter the api_key recognized as openai", what_keys(inputs)))
        yield from update_ui(chatbot=chatbot, history=history, msg="api_key has been imported") # Refresh interface
        return
    elif not is_any_api_key(chatbot._cookies['api_key']):
        chatbot.append((inputs, "api_key is missing. \n\n1. Temporary solution: Type api_key directly in the input area, then press Enter to submit. \n\n2. Long-term solution: configure in config.py."))
        yield from update_ui(chatbot=chatbot, history=history, msg="missing api_key") # Refresh interface
        return

    if additional_fn is not None:
        import core_functional
        importlib.reload(core_functional)    # hot update prompt
        core_functional = core_functional.get_core_functions()
        if "PreProcess" in core_functional[additional_fn]: inputs = core_functional[additional_fn]["PreProcess"](inputs)  # Get the preprocessing function (if any)
        inputs = core_functional[additional_fn]["Prefix"] + inputs + core_functional[additional_fn]["Suffix"]

    raw_input = inputs
    logging.info(f'[raw_input] {raw_input}')
    chatbot.append((inputs, ""))
    yield from updateUI(chatbot=chatbot, history=history, msg="waiting for response") # Refresh interface

    try:
        headers, payload = generate_payload(inputs, llm_kwargs, history, system_prompt, stream)
    except RuntimeError as e:
        chatbot[-1] = (inputs, f"The api-key you provided does not meet the requirements and does not contain any available api-keys for {llm_kwargs['llm_model']}. You may have selected the wrong model or request source.")
        yield from updateUI(chatbot=chatbot, history=history, msg="api-key does not meet the requirements") # Refresh interface
        return
        
    history.append(inputs); history.append("")

    retry = 0
    while True:
        try:
            # make a POST request to the API endpoint, stream=True
            from .bridge_all_llm import model_info
            endpoint = model_info[llm_kwargs['llm_model']]['endpoint']
            response = requests.post(endpoint, headers=headers, proxies=proxies,
                                    json=payload, stream=True, timeout=TIMEOUT_SECONDS);break
        except:
            retry += 1
            chatbot[-1] = ((chatbot[-1][0], timeout_bot_msg))
            retry_msg = f", retrying ({retry}/{MAX_RETRY}) ……" if MAX_RETRY > 0 else ""
            yield from update_ui(chatbot=chatbot, history=history, msg="Request timed out"+retry_msg) # Refresh interface
            if retry > MAX_RETRY: raise TimeoutError

    gpt_replying_buffer = ""
    
    is_head_of_the_stream = True
    if stream:
        stream_response =  response.iter_lines()
        while True:
            chunk = next(stream_response)
            # print(chunk.decode()[6:])
            if is_head_of_the_stream and (r'"object":"error"' not in chunk.decode()):
                # The first frame of the data stream does not carry content
                is_head_of_the_stream = False; continue
            
            if chunk:
                try:
                    chunk_decoded = chunk.decode()
                    # The former API2D
                    if ('data: [DONE]' in chunk_decoded) or (len(json.loads(chunk_decoded[6:])['choices'][0]["delta"]) == 0):
                        # It is judged as the end of the data stream, and gpt_replying_buffer is also written
                        logging.info(f'[response] {gpt_replying_buffer}')
                        break
                    # The body that handles the data stream
                    chunkjson = json.loads(chunk_decoded[6:])
                    status_text = f"finish_reason: {chunkjson['choices'][0]['finish_reason']}"
                    # If an exception is thrown here, it is generally because the text is too long, see the output of get_full_error for details
                    gpt_replying_buffer = gpt_replying_buffer + json.loads(chunk_decoded[6:])['choices'][0]["delta"]["content"]
                    history[-1] = gpt_replying_buffer
                    chatbot[-1] = (history[-2], history[-1])
                    yield from update_ui(chatbot=chatbot, history=history, msg=status_text) # Refresh interface

                except Exception as e:
                    traceback.print_exc()
                    yield from update_ui(chatbot=chatbot, history=history, msg="Json解析不合常规") # Refresh interface
                    chunk = get_full_error(chunk, stream_response)
                    chunk_decoded = chunk.decode()
                    error_msg = chunk_decoded
                    if "reduce the length" in error_msg:
                        if len(history) >= 2: history[-1] = ""; history[-2] = "" # Clear the current overflow input: history[-2] is this input, history[-1] is this output
                        history = clip_history(inputs=inputs, history=history, tokenizer=model_info[llm_kwargs['llm_model']]['tokenizer'], 
                                               max_token_limit=(model_info[llm_kwargs['llm_model']]['max_token'])) # release at least half of history
                        chatbot[-1] = (chatbot[-1][0], "[Local Message] Reduce the length. The input is too long this time, or the historical data is too long. The historical cache data has been partially released, you can try again. (If it fails again, it is more likely because the input is too long.)")
                        # history = []    # clear history
                    elif "does not exist" in error_msg:
                        chatbot[-1] = (chatbot[-1][0], f"[Local Message] Model {llm_kwargs['llm_model']} does not exist.")
                    elif "Incorrect API key" in error_msg:
                        chatbot[-1] = (chatbot[-1][0], "[Local Message] Incorrect API key.")
                    elif "exceeded your current quota" in error_msg:
                        chatbot[-1] = (chatbot[-1][0], "[Local Message] You exceeded your current quota.")
                    elif "bad forward key" in error_msg:
                        chatbot[-1] = (chatbot[-1][0], "[Local Message] Bad forward key.")
                    elif "Not enough point" in error_msg:
                        chatbot[-1] = (chatbot[-1][0], "[Local Message] Not enough point.")
                    else:
                        from toolbox import regular_txt_to_markdown
                        tb_str = '```\n' + trimmed_format_exc() + '```'
                        chatbot[-1] = (chatbot[-1][0], f"[Local Message] abnormal \n\n{tb_str} \n\n{regular_txt_to_markdown(chunk_decoded[4:])}")
                    yield from update_ui(chatbot=chatbot, history=history, msg="Json exception" + error_msg) # Refresh interface
                    return


def predict_no_ui_long_connection(inputs, llm_kwargs, history=[], sys_prompt="", observe_window=None, console_slience=False):
    """
     Send to chatGPT, wait for reply, complete in one go, no intermediate process will be displayed. But the method of stream is used internally to avoid the network cable being pinched in the middle.
     inputs:
         is the input for this query
     sys_prompt:
         System silent prompt
     llm_kwargs:
         Internal tuning parameters of chatGPT
     history:
         is a list of previous conversations
     observe_window = None:
         It is responsible for passing the output part across threads. Most of the time, it is only for the fancy visual effect, and it can be left blank. observe_window[0]: observation window. observe_window[1]: watchdog
    """
    watch_dog_patience = 5 # Patience of the watchdog, set it for 5 seconds
    headers, payload = generate_payload(inputs, llm_kwargs, history, system_prompt=sys_prompt, stream=True)
    retry = 0
    while True:
        try:
            # make a POST request to the API endpoint, stream=False
            from .bridge_all_llm import model_info
            endpoint = model_info[llm_kwargs['llm_model']]['endpoint']
            response = requests.post(endpoint, headers=headers, proxies=proxies,
                                    json=payload, stream=True, timeout=TIMEOUT_SECONDS); break
        except requests.exceptions.ReadTimeout as e:
            retry += 1
            traceback.print_exc()
            if retry > MAX_RETRY: raise TimeoutError
            if MAX_RETRY!=0: print(f'Request timed out, retrying({retry}/{MAX_RETRY}) ……')

    stream_response =  response.iter_lines()
    result = ''
    while True:
        try: chunk = next(stream_response).decode()
        except StopIteration: 
            break
        except requests.exceptions.ConnectionError:
            chunk = next(stream_response).decode() # Failed, try again? There is no other way to fail.
        if len(chunk)==0: continue
        if not chunk.startswith('data:'): 
            error_msg = get_full_error(chunk.encode('utf8'), stream_response).decode()
            if "reduce the length" in error_msg:
                raise ConnectionAbortedError("OpenAI rejected the request:" + error_msg)
            else:
                raise RuntimeError("OpenAI rejected the request:" + error_msg)
        if ('data: [DONE]' in chunk): break # api2d completed normally
        json_data = json.loads(chunk.lstrip('data:'))['choices'][0]
        delta = json_data["delta"]
        if len(delta) == 0: break
        if "role" in delta: continue
        if "content" in delta: 
            result += delta["content"]
            if not console_slience: print(delta["content"], end='')
            if observe_window is not None: 
                # Observation window to display the acquired data
                if len(observe_window) >= 1: observe_window[0] += delta["content"]
                # Watchdog, if the dog is not fed after the deadline, it will be terminated
                if len(observe_window) >= 2:  
                    if (time.time()-observe_window[1]) > watch_dog_patience:
                        raise RuntimeError("The user canceled the program.")
        else: raise RuntimeError("Unexpected Json structure:"+delta)
    if json_data['finish_reason'] == 'length':
        raise ConnectionAbortedError("It ends normally, but it shows insufficient Token, resulting in incomplete output. Please reduce the amount of text entered at a time.")
    return result