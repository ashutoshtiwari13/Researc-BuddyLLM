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

    #Handling markdown text formatting changes
    gr.Chatbot.postprocess = format_io

    from check_proxy import check_proxy, auto_update, warm_up_modules
    proxy_info = check_proxy(proxies)    

    gr_L1 = lambda: gr.Row().style()
    gr_L2 = lambda scale: gr.Column(scale=scale)
    if LAYOUT == "TOP-DOWN":
        gr_L1 = lambda: DummyWith()
        gr_L2 = lambda scale: gr.Row()
        CHATBOT_HEIGHT /= 2
    

    cancel_handles =[]
    with gr.Blocks(itle="Research Buddy LLM", analytics_enabled = False) as demo:
        gr.HTML(title_html)

        cookies = gr.State({'apy_key': API_KEY, 'llm_model':LLM_MODEL})
        with gr.L1():
            with gr.L2(scale=2):
                chatbot = gr.Chatbot(label = f"Current Model : {LLM_MODEL}")
                chatbot.style(height = CHATBOT_HEIGHT)
                history = gr.State([])
            with gr.L2(scale=1):
                with gr.Accordion("input area",open=True) as primary_input_area:
                    with gr.Row():
                        txt = gr.Textbox(show_label = False,placeholder = "Input Question here").style(container = False)
                    with gr.Row():
                        submitBtn = gr.Button("Submit", variant = "Primary")

                    with gr.Row():
                        resetBtn = gr.Button("Reset", variant = "Secondary"); resetBtn.style(size="sm")
                        StopBtn = gr.Button("Stop", variant = "Secondary");StopBtn.style(size="sm")
                        ClearBtn = gr.Button("Clear", variant = "Secondary", visible = False);ClearBtn.style(size="sm")

                    with gr.Row():
                        status = gr.Markdonw(f"Tip: Press Enter to submit, press Shift+Enter to wrap. Current model: {LLM_MODEL} \n {proxy_info}")

                        

                    



if __name__ == "__main__":
    main()    