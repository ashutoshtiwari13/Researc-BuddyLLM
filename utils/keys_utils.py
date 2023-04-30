import re
from latex2mathml.converter import convert as tex2mathml

def is_openai_api_key(key):
    API_MATCH = re.match(r"sk-[a-zA-Z0-9]{48}$", key)
    return bool(API_MATCH)

def is_api2d_key(key):
    if key.startswith('fk') and len(key) == 41:
        return True
    else:
        return False

def is_any_api_key(key):
    if ',' in key:
        keys = key.split(',')
        for k in keys:
            if is_any_api_key(k): return True
        return False
    else:
        return is_openai_api_key(key) or is_api2d_key(key)

def what_keys(keys):
    avail_key_list = {'OpenAI Key':0, "API2D Key":0}
    key_list = keys.split(',')

    for k in key_list:
        if is_openai_api_key(k): 
            avail_key_list['OpenAI Key'] += 1

    for k in key_list:
        if is_api2d_key(k): 
            avail_key_list['API2D Key'] += 1

    return f"Detected： OpenAI Key {avail_key_list['OpenAI Key']} 个，API2D Key {avail_key_list['API2D Key']}"

def select_api_key(keys, llm_model):
    import random
    avail_key_list = []
    key_list = keys.split(',')

    if llm_model.startswith('gpt-'):
        for k in key_list:
            if is_openai_api_key(k): avail_key_list.append(k)

    if llm_model.startswith('api2d-'):
        for k in key_list:
            if is_api2d_key(k): avail_key_list.append(k)

    if len(avail_key_list) == 0:
        raise RuntimeError(f"The api-key provided does not meet requirements key error : {llm_model}wrong model or key chosen")

    api_key = random.choice(avail_key_list) 
    return api_key