from utils.functional_utils import updateUI,CatchException
from utils.process_utils import report_exception, write_results_to_file
fast_debug = True


class JupyterNotebookGroup():
    def __init__(self):
        self.file_paths = []
        self.file_contents = []
        self.sp_file_contents = []
        self.sp_file_index = []
        self.sp_file_tag = []

        from core_llm.bridge_all_llm import model_info
        enc = model_info["gpt-3.5-turbo"]['tokenizer']
        def get_token_num(txt): return len(
            enc.encode(txt, disallowed_special=()))
        self.get_token_num = get_token_num


    def run_file_split(self,max_token_limit=1900):
        """
        Preprocess long texts
        """
        for index,file_content in enumerate(self.file_contents):
                if self.get_token_num(file_content) < max_token_limit:
                self.sp_file_contents.append(file_content)
                self.sp_file_index.append(index)
                self.sp_file_tag.append(self.file_paths[index])
            else:
                from .core_utils import breakdown_txt_to_satisfy_token_limit_for_pdf
                segments = breakdown_txt_to_satisfy_token_limit_for_pdf(
                    file_content, self.get_token_num, max_token_limit)
                for j, segment in enumerate(segments):
                    self.sp_file_contents.append(segment)
                    self.sp_file_index.append(index)
                    self.sp_file_tag.append(
                        self.file_paths[index] + f".part-{j}.txt")




def parseNotebook(filename, enable_markdown=1):
    import json

    CodeBlocks = []
    with open(filename, 'r', encoding='utf-8', errors='replace') as f:
        notebook = json.load(f)
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and cell['source']:
            # remove blank lines
            cell['source'] = [line for line in cell['source'] if line.strip()
                              != '']
            CodeBlocks.append("".join(cell['source']))
        elif enable_markdown and cell['cell_type'] == 'markdown' and cell['source']:
            cell['source'] = [line for line in cell['source'] if line.strip()
                              != '']
            CodeBlocks.append("Markdown:"+"".join(cell['source']))

    Code = ""
    for idx, code in enumerate(CodeBlocks):
        Code += f"This is {idx+1}th code block: \n"
        Code += code+"\n"

    return Code 

            

def ipynbExplained(file_manifest, project_folder, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt):
    from .core_utils import request_gpt_model_multi_threads_with_very_awesome_ui_and_high_efficiency

    enable_markdown = plugin_kwargs.get("advanced_arg", "1")
    try:
        enable_markdown = int(enable_markdown)
    except ValueError:
        enable_markdown = 1

    pfg = JupyterNotebookGroup()

    for fp in file_manifest:
        file_content = parseNotebook(fp, enable_markdown=enable_markdown)
        pfg.file_paths.append(fp)
        pfg.file_contents.append(file_content)

    #  <-------- Split too long ipynb file---------->
    pfg.run_file_split(max_token_limit=1024)
    n_split = len(pfg.sp_file_contents)

    inputs_array = [r"This is a Jupyter Notebook file, tell me about Each Block in English. Focus Just On Code." +
                    r"If a block starts with `Markdown` which means it's a markdown block in ipynbipynb. " +
                    r"Start a new line for a block and block num use English." +
                    f"\n\n{frag}" for frag in pfg.sp_file_contents]
    inputs_show_user_array = [f"{f}The analysis is as follows" for f in pfg.sp_file_tag]
    sys_prompt_array = ["You are a professional programmer."] * n_split

    gpt_response_collection = yield from request_gpt_model_multi_threads_with_very_awesome_ui_and_high_efficiency(
        inputs_array=inputs_array,
        inputs_show_user_array=inputs_show_user_array,
        llm_kwargs=llm_kwargs,
        chatbot=chatbot,
        history_array=[[""] for _ in range(n_split)],
        sys_prompt_array=sys_prompt_array,
        # max_workers=5,  # max parallel load allowed
        scroller_max_len=80
    )

    #  <-------- Sort the results, And Exit ---------->
    block_result = "  \n".join(gpt_response_collection)
    chatbot.append(("The results of the analysis are as follows", block_result))
    history.extend(["The results of the analysis are as follows", block_result])
    yield from updateUI(chatbot=chatbot, history=history)  # Refresh Interface

    #  <-------- Write to file, And Exit ---------->
    res = write_results_to_file(history)
    chatbot.append(("Finished?", res))
    yield from updateUI(chatbot=chatbot, history=history) # Refresh Interface


@CatchException
def parsingIpynbFiles(txt, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt, web_port):
    chatbot.append([
        "Function plugin function?",
        "Parse the IPynb file."])
    yield from updateUI(chatbot=chatbot, history=history)  # Refresh Interface

    history = []    # clear history
    import glob
    import os
    if os.path.exists(txt):
        project_folder = txt
    else:
        if txt == "":
            txt = 'Empty input field'
        report_execption(chatbot, history,
                         a=f"parse project: {txt}", b=f"Could not find local project or do not have permission to access: {txt}")
        yield from updateUI(chatbot=chatbot, history=history)  # Refresh Interface
        return
    if txt.endswith('.ipynb'):
        file_manifest = [txt]
    else:
        file_manifest = [f for f in glob.glob(
            f'{project_folder}/**/*.ipynb', recursive=True)]
    if len(file_manifest) == 0:
        report_execption(chatbot, history,
                         a=f"parse project: {txt}", b=f"Could not find any .ipynb files: {txt}")
        yield from updateUI(chatbot=chatbot, history=history)  # Refresh Interface
        return
    yield from ipynbExplained(file_manifest, project_folder, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt, )
