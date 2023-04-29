#Avoid Accidental Pollution of Proxy Networks
import os; os.environ['no_proxy'] = '*'

def main():
    import gradio as gr
    #try gradio for a example
    # def greet(name):
    #     return "Hello" + name + "!"
    # demo = gr.Interface(fn = greet, inputs = "text", outputs = "text")
    # demo.launch()

    from core_llm.bridge_all_llm import predict
    from utils.functional_utils import get_conf, GenericArgsWrapper
    from utils.process_utils import format_io,find_free_port, on_file_uploaded, on_report_generated, DummyWith

    proxies, WEB_PORT, LLM_MODEL, CONCURRENT_COUNT, AUTHENTICATION, CHATBOT_HEIGHT, LAYOUT, API_KEY, AVAIL_LLM_MODELS = \
        get_conf('proxies', 'WEB_PORT', 'LLM_MODEL', 'CONCURRENT_COUNT', 'AUTHENTICATION', 'CHATBOT_HEIGHT', 'LAYOUT', 'API_KEY', 'AVAIL_LLM_MODELS')    


    #If WEB_PORT is -1, the WEB port is randomly selected
    PORT = find_free_port() if WEB_PORT <= 0 else WEB_PORT
    if not AUTHENTICATION: AUTHENTICATION = None    

    from utils.check_proxy import get_current_version
    initial_prompt = "Serve me as a writing and programming assistant."
    title_html = f"<h1 align=\"center\">Research Buddy LLM{get_current_version()}</h1>"
    description =  """Your one-stop solution buddy for optimized and faster Research"""

    # logging runtime errors into ./conversation_log/ directory
    import logging
    os.makedirs("conversation_log", exist_ok=True)
    try:logging.basicConfig(filename="conversation_log/chat_secrets.log", level=logging.INFO, encoding="utf-8")
    except:logging.basicConfig(filename="conversation_log/chat_secrets.log", level=logging.INFO)
    print("All inquiry records will be automatically saved in the local directory ./conversation_log/chat_secrets.log, please pay attention to self-privacy protection!")

    #get Core prompt function generator
    from core_llm_functional import get_core_functions
    functional = get_core_functions()

    #get Custom applications
    from core_tools import get_core_tools
    tools = get_core_tools()

    



if __name__ == "__main__":
    main()    