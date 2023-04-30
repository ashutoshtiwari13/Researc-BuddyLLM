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
    from core_llm_functional import get_core_llm_functions
    functional = get_core_llm_functions()

    #get Custom applications
    from core_tools import get_core_tools
    tools = get_core_tools()

    #Handling markdown text formatting changes
    gr.Chatbot.postprocess = format_io

    from utils.check_proxy import check_proxy, auto_update, warm_up_modules
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
        with gr_L1():
            with gr_L2(scale=2):
                chatbot = gr.Chatbot(label = f"Current Model : {LLM_MODEL}")
                chatbot.style(height = CHATBOT_HEIGHT)
                history = gr.State([])
            with gr_L2(scale=1):
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
                        status = gr.Markdown(f"Tip: Press Enter to submit, press Shift+Enter to wrap. Current model: {LLM_MODEL} \n {proxy_info}")

                with gr.Accordion("Basic functional area", open=True) as area_basic_fn:
                    with gr.Row():
                        for k in functional:
                            variant = functional[k]["Color"] if "Color" in functional[k] else "secondary"
                            functional[k]["Button"] = gr.Button(k, variant=variant)

                with gr.Accordion("Tools functional area", open=True) as area_tool_fn:
                    with gr.Row():
                        gr.Markdown("Note: The function plug-ins identified by the \"red color\" below need to read the path from the input area as a parameter.")
                    with gr.Row():
                        for k in tools:
                            if not tools[k].get("AsButton", True): continue
                            variant = tools[k]["Color"] if "Color" in tools[k] else "secondary"
                            tools[k]["Button"] = gr.Button(k, variant=variant)
                            tools[k]["Button"].style(size="sm")                            
                    with gr.Row():
                        with gr.Accordion("More function plugins", open=True):
                            dropdown_fn_list = [k for k in tools.keys() if not tools[k].get("AsButton", True)]
                            with gr.Row():
                                dropdown = gr.Dropdown(dropdown_fn_list, value=r"Open plugin list", label="").style(container=False)
                            with gr.Row():
                                plugin_advanced_arg = gr.Textbox(show_label=True, label="Advanced parameter input area", visible=False, 
                                                                 placeholder="Here is the advanced parameter input area for the special function plugin").style(container=False)
                            with gr.Row():
                                switchy_bt = gr.Button(r"Please select from the plugin list first", variant="secondary")


                    with gr.Row():
                        with gr.Accordion("Click to expand the \"File Upload Area\". Uploading local files can be called by the red function plugin.", open=False) as area_file_upload:
                            file_upload = gr.Files(label="Any file, but uploading compressed files is recommended(zip, tar)", file_count="multiple")

                
                            




        def fn_area_visibility(a):
            ret = {}
            ret.update({area_basic_fn : gr.update(visible =("Basic functional area" in a ))})

    #Gradio's inbrowser trigger is not stable, roll back the code to the original browser opening function
    def auto_opentab_delay():
        import threading, webbrowser, time
        print(f"If your browser doesn't open automatically, copy and go to the following URL:")
        print(f"\t（bright theme): http://localhost:{PORT}")
        print(f"\t（dark theme): http://localhost:{PORT}/?__dark-theme=true")
        def open():
            time.sleep(2)       # open browser
            DARK_MODE, = get_conf('DARK_MODE')
            if DARK_MODE: webbrowser.open_new_tab(f"http://localhost:{PORT}/?__dark-theme=true")
            else: webbrowser.open_new_tab(f"http://localhost:{PORT}")
        threading.Thread(target=open, name="open-browser", daemon=True).start()
        threading.Thread(target=auto_update, name="self-upgrade", daemon=True).start()
        threading.Thread(target=warm_up_modules, name="warm-up", daemon=True).start()   

    auto_opentab_delay()
    demo.queue(concurrency_count=CONCURRENT_COUNT).launch(server_name="0.0.0.0", server_port=PORT, auth=AUTHENTICATION, favicon_path="logo.png")        


    # If you need to run under the secondary path
    # CUSTOM_PATH, = get_conf('CUSTOM_PATH')
    # if CUSTOM_PATH != "/": 
    #     from toolbox import run_gradio_in_subpath
    #     run_gradio_in_subpath(demo, auth=AUTHENTICATION, port=PORT, custom_path=CUSTOM_PATH)
    # else: 
    #     demo.launch(server_name="0.0.0.0", server_port=PORT, auth=AUTHENTICATION, favicon_path="docs/logo.png")                



if __name__ == "__main__":
    main()    