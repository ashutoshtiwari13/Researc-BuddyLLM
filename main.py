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
                with gr.Accordion("input area",open=True) as area_input_primary:
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

                with gr.Accordion("Tools functional area", open=True) as area_tools_fn:
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

                with gr.Accordion("Change Model & SysPrompt & Interface Layout", open=(LAYOUT == "TOP-DOWN")):
                    system_prompt = gr.Textbox(show_label=True, placeholder=f"System Prompt", label="System prompt", value=initial_prompt)
                    top_p = gr.Slider(minimum=-0, maximum=1.0, value=1.0, step=0.01,interactive=True, label="Top-p (nucleus sampling)",)
                    temperature = gr.Slider(minimum=-0, maximum=2.0, value=1.0, step=0.01, interactive=True, label="Temperature",)
                    max_length_sl = gr.Slider(minimum=256, maximum=4096, value=512, step=1, interactive=True, label="Local LLM MaxLength",)
                    checkboxes = gr.CheckboxGroup(["Basic function area", "Function plug-in area", "Bottom input area", "Input clear key", "Plug-in parameter area"], value=["Basic function area", "Function plug-in area"], label = "Show/Hide Ribbon")
                    md_dropdown = gr.Dropdown(AVAIL_LLM_MODELS, value=LLM_MODEL, label="Replace LLM model/request source").style(container=False)  

                    gr.Markdown(description)

                with gr.Accordion("Alternate input field",open = True, visible= False) as area_input_secondary:
                    with gr.Row():
                        txt2 = gr.Textbox(show_label=False, placeholder="Input question here.", label="input area 2").style(container=False)
                    with gr.Row():
                        submitBtn2 = gr.Button("submit", variant="primary")    
                    with gr.Row():
                        resetBtn2 = gr.Button("Reset", variant="secondary"); resetBtn2.style(size="sm")
                        stopBtn2 = gr.Button("Stop", variant="secondary"); stopBtn2.style(size="sm")
                        clearBtn2 = gr.Button("Clear", variant="secondary", visible=False); clearBtn2.style(size="sm")           

        
         #Ribbon Display Switch Interaction with Ribbon

        def fn_area_visibility(a):
            ret = {}
            ret.update({area_basic_fn : gr.update(visible =("Basic functional area" in a ))})
            ret.update({area_tools_fn: gr.update(visible=("Tools Functions Area" in a))})
            ret.update({area_input_primary: gr.update(visible=("Primary input area" not in a))})
            ret.update({area_input_secondary: gr.update(visible=("Secondary input area" in a))})
            ret.update({clearBtn: gr.update(visible=("input clear key" in a))})
            ret.update({clearBtn2: gr.update(visible=("input clear key" in a))})
            ret.update({plugin_advanced_arg: gr.update(visible=("Plug-in parameter area" in a))})
            if "bottom input area" in a: ret.update({txt: gr.update(value="")})
            return ret
        
        checkboxes.select(fn_area_visibility, [checkboxes], [area_basic_fn, area_tools_fn, area_input_primary ,area_input_secondary, txt, txt2, clearBtn, clearBtn2, plugin_advanced_arg] )   

        #Clean up recurring control handle combinations
        input_combo = [cookies, max_length_sl, md_dropdown, txt, txt2, top_p, temperature, chatbot, history, system_prompt, plugin_advanced_arg]
        output_combo = [cookies, chatbot, history, status]
        #calling the prediction function
        predict_args = dict(fn=GenericArgsWrapper(predict), inputs=input_combo, outputs=output_combo)

        #submit button, reset button
        cancel_handles.append(txt.submit(**predict_args))
        cancel_handles.append(txt2.submit(**predict_args))
        cancel_handles.append(submitBtn.click(**predict_args))
        cancel_handles.append(submitBtn2.click(**predict_args))
        resetBtn.click(lambda: ([], [], "reset"), None, [chatbot, history, status])
        resetBtn2.click(lambda: ([], [], "reset"), None, [chatbot, history, status])
        clearBtn.click(lambda: ("",""), None, [txt, txt2])
        clearBtn2.click(lambda: ("",""), None, [txt, txt2])  

        #Callback function registration of the basic functional area
        for k in functional:
            click_handle = functional[k]["Button"].click(fn=GenericArgsWrapper(predict), inputs=[*input_combo, gr.State(True), gr.State(k)], outputs=output_combo)
            cancel_handles.append(click_handle)      


        file_upload.upload(on_file_uploaded, [file_upload, chatbot, txt, txt2, checkboxes], [chatbot, txt, txt2])   

        #Function plugin - fixed button area
        for k in tools:
            if not tools[k].get("AsButton", True): continue
            click_handle = tools[k]["Button"].click(GenericArgsWrapper(tools[k]["Function"]), [*input_combo, gr.State(PORT)], output_combo)
            click_handle.then(on_report_generated, [file_upload, chatbot], [file_upload, chatbot])
            cancel_handles.append(click_handle)    

        #Function plugin - interaction between drop-down menu and variable button
        def on_dropdown_changed(k):
            variant = tools[k]["Color"] if "Color" in tools[k] else "secondary"
            ret = {switchy_bt: gr.update(value=k, variant=variant)}
            if tools[k].get("AdvancedArgs", False): # Whether to invoke the advanced plug-in parameter area
                ret.update({plugin_advanced_arg: gr.update(visible=True,  label=f"插Explanation of advanced parameters for piece[{k}]:" + tools[k].get("ArgsReminder", [f"No description of advanced parameters is provided"]))})
            else:
                ret.update({plugin_advanced_arg: gr.update(visible=False, label=f"Plugins[{k}] do not require advanced parameters.")})
            return ret
        dropdown.select(on_dropdown_changed, [dropdown], [switchy_bt, plugin_advanced_arg] )

        def on_md_dropdown_changed(k):
            return {chatbot: gr.update(label="Current model:"+k)}
        md_dropdown.select(on_md_dropdown_changed, [md_dropdown], [chatbot] )


        # The callback function registration of the variable button
        def route(k, *args, **kwargs):
            if k in [r"Open the plugin list", r"Please select from the plugin list first"]: return
            yield from GenericArgsWrapper(tools[k]["Function"])(*args, **kwargs)
        click_handle = switchy_bt.click(route,[switchy_bt, *input_combo, gr.State(PORT)], output_combo)
        click_handle.then(on_report_generated, [file_upload, chatbot], [file_upload, chatbot])
        cancel_handles.append(click_handle)

        # Callback function registration of the terminate button
        stopBtn.click(fn=None, inputs=None, outputs=None, cancels=cancel_handles)
        stopBtn2.click(fn=None, inputs=None, outputs=None, cancels=cancel_handles)        


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