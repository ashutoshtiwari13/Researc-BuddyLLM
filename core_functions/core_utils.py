from utils.functional_utils import updateUI, get_conf, trimmed_format_exc

def input_clipping(inputs, history, max_token_limit):
    import numpy as np
    from _llm.bridge_all_llm import model_info
    enc = model_info["gpt-3.5-turbo"]['tokenizer']
    def get_token_num(txt): return len(enc.encode(txt, disallowed_special=()))

    mode = 'input-and-history'
    # When the proportion of tokens in the input part is less than half of the full text, only the history is cropped
    input_token_num = get_token_num(inputs)
    if input_token_num < max_token_limit//2: 
        mode = 'only-history'
        max_token_limit = max_token_limit - input_token_num

    everything = [inputs] if mode == 'input-and-history' else ['']
    everything.extend(history)
    n_token = get_token_num('\n'.join(everything))
    everything_token = [get_token_num(e) for e in everything]
    delta = max(everything_token) // 16 # Granularity at truncation
        
    while n_token > max_token_limit:
        where = np.argmax(everything_token)
        encoded = enc.encode(everything[where], disallowed_special=())
        clipped_encoded = encoded[:len(encoded)-delta]
        everything[where] = enc.decode(clipped_encoded)[:-1]    # -1 to remove the may-be illegal char
        everything_token[where] = get_token_num(everything[where])
        n_token = get_token_num('\n'.join(everything))

    if mode == 'input-and-history':
        inputs = everything[0]
    else:
        pass
    history = everything[1:]
    return inputs, history


def request_gpt_model_in_new_thread_with_ui_alive(
        inputs, inputs_show_user, llm_kwargs, 
        chatbot, history, sys_prompt, refresh_interval=0.2,
        handle_token_exceed=True, 
        retry_times_at_unknown_error=2,
        ):
    """
     Request GPT model, request the GPT model while keeping the user interface active.

     Input parameter Args (the input variables ending with _array are lists, and the length of the list is the number of subtasks. When executing, the list will be disassembled and placed in each sub-thread for execution):
         inputs (string): List of inputs (inputs)
         inputs_show_user (string): List of inputs to show user (the input displayed in the report, with the help of this parameter, the long-winded real input is hidden in the summary report to enhance the readability of the report)
         top_p (float): Top p value for sampling from model distribution (GPT parameter, floating point number)
         temperature (float): Temperature value for sampling from model distribution (GPT parameter, floating point number)
         chatbot: chatbot inputs and outputs (user interface dialog window handle, used for data flow visualization)
         history (list): List of chat history (history, list of chat history)
         sys_prompt (string): List of system prompts (system input, list, premise prompts for input to GPT, such as how you are a translator)
         refresh_interval (float, optional): Refresh interval for UI (default: 0.2) (refresh interval frequency, it is recommended to be lower than 1, not higher than 3, only for visual effects)
         handle_token_exceed: Whether to automatically handle token overflow, if you choose to handle automatically, it will be violently truncated when it overflows, and it is enabled by default
         retry_times_at_unknown_error: number of retries on failure

     Output Returns:
         future: output, the result returned by GPT
    """
    import time
    from concurrent.futures import ThreadPoolExecutor
    from core_llm.bridge_all_llm import predict_no_ui_long_connection
    # Customer feedback
    chatbot.append([inputs_show_user, ""])
    yield from updateUI(chatbot=chatbot, history=[]) # Refresh interface
    executor = ThreadPoolExecutor(max_workers=16)
    mutable = ["", time.time(), ""]
    def _req_gpt(inputs, history, sys_prompt):
        retry_op = retry_times_at_unknown_error
        exceeded_cnt = 0
        while True:
            # watchdog error
            if len(mutable) >= 2 and (time.time()-mutable[1]) > 5: 
                raise RuntimeError("Program termination detected.")
            try:
                # [Case 1]: Completed successfully
                result = predict_no_ui_long_connection(
                    inputs=inputs, llm_kwargs=llm_kwargs,
                    history=history, sys_prompt=sys_prompt, observe_window=mutable)
                return result
            except ConnectionAbortedError as token_exceeded_error:
                # [Second case]: Token overflow
                if handle_token_exceed:
                    exceeded_cnt += 1
                    # [Selection processing] Try to calculate the ratio and keep as much text as possible
                    from toolbox import get_reduce_token_percent
                    p_ratio, n_exceed = get_reduce_token_percent(str(token_exceeded_error))
                    MAX_TOKEN = 4096
                    EXCEED_ALLO = 512 + 512 * exceeded_cnt
                    inputs, history = input_clipping(inputs, history, max_token_limit=MAX_TOKEN-EXCEED_ALLO)
                    mutable[0] += f'[Local Message] Warning, if the text is too long, it will be truncated, and the Token overflow number:{n_exceed}。\n\n'
                    continue # return to retry
                else:
                    # [Choose to give up]
                    tb_str = '```\n' + trimmed_format_exc() + '```'
                    mutable[0] += f"[Local Message] Warning, encountered a problem during execution, Traceback:\n\n{tb_str}\n\n"
                    return mutable[0] # give up
            except:
                # [Third case]: other errors: retry several times
                tb_str = '```\n' + trimmed_format_exc() + '```'
                print(tb_str)
                mutable[0] += f"[Local Message] Warning, encountered a problem during execution, Traceback:\n\n{tb_str}\n\n"
                if retry_op > 0:
                    retry_op -= 1
                    mutable[0] += f"[Local Message] 重试中，请稍等 {retry_times_at_unknown_error-retry_op}/{retry_times_at_unknown_error}：\n\n"
                    if ("Rate limit reached" in tb_str) or ("Too Many Requests" in tb_str):
                        time.sleep(30)
                    time.sleep(5)
                    continue # return to retry
                else:
                    time.sleep(5)
                    return mutable[0] # give up

    # Submit the task
    future = executor.submit(_req_gpt, inputs, history, sys_prompt)
    while True:
        # yield一times to refresh the front-end page
        time.sleep(refresh_interval)
        # "Feed the Dog" (watchdog)
        mutable[1] = time.time()
        if future.done():
            break
        chatbot[-1] = [chatbot[-1][0], mutable[0]]
        yield from update_ui(chatbot=chatbot, history=[]) # Refresh interface

    final_result = future.result()
    chatbot[-1] = [chatbot[-1][0], final_result]
    yield from update_ui(chatbot=chatbot, history=[]) # If it succeeds in the end, delete the error message
    return final_result


def request_gpt_model_multi_threads_with_very_awesome_ui_and_high_efficiency(
        inputs_array, inputs_show_user_array, llm_kwargs, 
        chatbot, history_array, sys_prompt_array, 
        refresh_interval=0.2, max_workers=-1, scroller_max_len=30,
        handle_token_exceed=True, show_user_at_complete=False,
        retry_times_at_unknown_error=2,
        ):
    """
    Request GPT model using multiple threads with UI and high efficiency
     Request the [multithreaded] version of the GPT model.
     It has the following functions:
         Feedback remote data stream on UI in real time
         Using the thread pool, the size of the thread pool can be adjusted to avoid openai's flow limit error
         Handling mid-stop situations
         When there is a problem with the network, etc., the traceback and the received data will be transferred to the output

     Input parameter Args (the input variables ending with _array are lists, and the length of the list is the number of subtasks. When executing, the list will be disassembled and placed in each sub-thread for execution):
         inputs_array (list): List of inputs (inputs for each subtask)
         inputs_show_user_array (list): List of inputs to show user (the input of each subtask displayed in the report, with the help of this parameter, the long-winded real input is hidden in the summary report, and the readability of the report is enhanced)
         llm_kwargs: llm_kwargs parameter
         chatbot: chatbot (UI dialog window handle, used for data flow visualization)
         history_array (list): List of chat history (historical dialogue input, double-layer list, the first-level list is the subtask decomposition, and the second-level list is the dialogue history)
         sys_prompt_array (list): List of system prompts (system input, list, premise prompts for input to GPT, such as how you are a translator)
         refresh_interval (float, optional): Refresh interval for UI (default: 0.2) (refresh interval frequency, it is recommended to be lower than 1, not higher than 3, only for visual effects)
         max_workers (int, optional): Maximum number of threads (default: see config.py) (the maximum number of threads, if there are too many subtasks, you need to use this option to prevent frequent requests to openai from causing errors)
         scroller_max_len (int, optional): Maximum length for scroller (default: 30) (the number of characters received at the end of the display of the data stream is only for visual effects)
         handle_token_exceed (bool, optional): (whether to automatically reduce the text when the input is too long)
         handle_token_exceed: Whether to automatically handle token overflow, if you choose to handle automatically, it will be violently truncated when it overflows, and it is enabled by default
         show_user_at_complete (bool, optional): (at the end, show the complete input-output result in the chat box)
         retry_times_at_unknown_error: The number of retries when subtasks fail

     Output Returns:
         list: List of GPT model responses (The output of each subtask is summarized. If a subtask fails, the response will carry traceback error information, which is convenient for debugging and locating the problem.
    """
    import time, random
    from concurrent.futures import ThreadPoolExecutor
    from request_llm.bridge_all_llm import predict_no_ui_long_connection
    assert len(inputs_array) == len(history_array)
    assert len(inputs_array) == len(sys_prompt_array)
    if max_workers == -1: # read configuration file
        try: max_workers, = get_conf('DEFAULT_WORKER_NUM')
        except: max_workers = 8
        if max_workers <= 0: max_workers = 3
    # Blocking chatglm's multi-threading may cause serious lag
    if not (llm_kwargs['llm_model'].startswith('gpt-') or llm_kwargs['llm_model'].startswith('api2d-')):
        max_workers = 1
        
    executor = ThreadPoolExecutor(max_workers=max_workers)
    n_frag = len(inputs_array)
    # customer feedback
    chatbot.append(["Please start multi-threaded operation.", ""])
    yield from update_ui(chatbot=chatbot, history=[]) # Refresh interface
    # passing across threads
    mutable = [["", time.time(), "Waiting"] for _ in range(n_frag)]

    # child thread task
    def _req_gpt(index, inputs, history, sys_prompt):
        gpt_say = ""
        retry_op = retry_times_at_unknown_error
        exceeded_cnt = 0
        mutable[index][2] = "in execution"
        while True:
            # watchdog error
            if len(mutable[index]) >= 2 and (time.time()-mutable[index][1]) > 5: 
                raise RuntimeError("Program termination detected.")
            try:
                # 【Case 1]: successfully completed
                # time.sleep(10); raise RuntimeError("test")
                gpt_say = predict_no_ui_long_connection(
                    inputs=inputs, llm_kwargs=llm_kwargs, history=history, 
                    sys_prompt=sys_prompt, observe_window=mutable[index], console_slience=True
                )
                mutable[index][2] = "succeeded"
                return gpt_say
            except ConnectionAbortedError as token_exceeded_error:
                # [Second case]: Token overflow,
                if handle_token_exceed:
                    exceeded_cnt += 1
                    # [Selection processing] Try to calculate the ratio and keep as much text as possible
                    from utils.process_utils import get_token_usage_data
                    p_ratio, n_exceed = get_token_usage_data(str(token_exceeded_error))
                    MAX_TOKEN = 4096
                    EXCEED_ALLO = 512 + 512 * exceeded_cnt
                    inputs, history = input_clipping(inputs, history, max_token_limit=MAX_TOKEN-EXCEED_ALLO)
                    gpt_say += f'[Local Message] Warning, if the text is too long, it will be truncated, and the Token overflow number:{n_exceed}。\n\n'
                    mutable[index][2] = f"Truncate retry"
                    continue # return to retry
                else:
                    # 【Choose to give up】
                    tb_str = '```\n' + trimmed_format_exc() + '```'
                    gpt_say += f"[Local Message] Warning, thread {index} encountered a problem during execution, Traceback:\n\n{tb_str}\n\n"
                    if len(mutable[index][0]) > 0: gpt_say += "Answers received before this thread failed:\n\n" + mutable[index][0]
                    mutable[index][2] = "Input too long and discarded"
                    return gpt_say # give up
            except:
                # [Third case]: other errors
                tb_str = '```\n' + trimmed_format_exc() + '```'
                print(tb_str)
                gpt_say += f"[Local Message] Warning, thread {index} encountered a problem during execution, Traceback:\n\n{tb_str}\n\n"
                if len(mutable[index][0]) > 0: gpt_say += "Answers received before this thread failed:\n\n" + mutable[index][0]
                if retry_op > 0:
                    retry_op -= 1
                    wait = random.randint(5, 20)
                    if ("Rate limit reached" in tb_str) or ("Too Many Requests" in tb_str):
                        wait = wait * 3
                        fail_info = "OpenAI binding credit card can lift the frequency limit "
                    else:
                        fail_info = ""
                    # Maybe wait a dozen seconds and things will get better
                    for i in range(wait):
                        mutable[index][2] = f"{fail_info}wait for retry {wait-i}"; time.sleep(1)
                    # start retrying
                    mutable[index][2] = f" retrying {retry_times_at_unknown_error-retry_op}/{retry_times_at_unknown_error}"
                    continue # return to retry
                else:
                    mutable[index][2] = "failed"
                    wait = 5
                    time.sleep(5)
                    return gpt_say # give up

    # Asynchronous task started
    futures = [executor.submit(_req_gpt, index, inputs, history, sys_prompt) for index, inputs, history, sys_prompt in zip(
        range(len(inputs_array)), inputs_array, history_array, sys_prompt_array)]
    cnt = 0
    while True:
        # yield一times to refresh the front-end page
        time.sleep(refresh_interval)
        cnt += 1
        worker_done = [h.done() for h in futures]
        if all(worker_done):
            executor.shutdown()
            break
        # Better UI Visual Effects
        observe_win = []
        # Each thread has to "feed the dog" (watchdog)
        for thread_index, _ in enumerate(worker_done):
            mutable[thread_index][1] = time.time()
        # Print something fun on the front end
        for thread_index, _ in enumerate(worker_done):
            print_something_really_funny = "[ ...`"+mutable[thread_index][0][-scroller_max_len:].\
                replace('\n', '').replace('```', '...').replace(
                    ' ', '.').replace('<br/>', '.....').replace('$', '.')+"`... ]"
            observe_win.append(print_something_really_funny)
        # Print something fun on the front end
        stat_str = ''.join([f'`{mutable[thread_index][2]}`: {obs}\n\n' 
                            if not done else f'`{mutable[thread_index][2]}`\n\n' 
                            for thread_index, done, obs in zip(range(len(worker_done)), worker_done, observe_win)])
        # Print something fun on the front end的东西
        chatbot[-1] = [chatbot[-1][0], f'The multi-threaded operation has started and completed: \n\n{stat_str}' + ''.join(['.']*(cnt % 10+1))]
        yield from update_ui(chatbot=chatbot, history=[]) # refresh interface
    
    # Asynchronous task ends
    gpt_response_collection = []
    for inputs_show_user, f in zip(inputs_show_user_array, futures):
        gpt_res = f.result()
        gpt_response_collection.extend([inputs_show_user, gpt_res])
    
    # Whether to display the result on the interface at the end
    if show_user_at_complete:
        for inputs_show_user, f in zip(inputs_show_user_array, futures):
            gpt_res = f.result()
            chatbot.append([inputs_show_user, gpt_res])
            yield from update_ui(chatbot=chatbot, history=[]) # refresh interface
            time.sleep(0.3)
    return gpt_response_collection



def breakdown_txt_to_satisfy_token_limit(txt, get_token_fn, limit):
    def cut(txt_tocut, must_break_at_empty_line):  # recursion
        if get_token_fn(txt_tocut) <= limit:
            return [txt_tocut]
        else:
            lines = txt_tocut.split('\n')
            estimated_line_cut = limit / get_token_fn(txt_tocut) * len(lines)
            estimated_line_cut = int(estimated_line_cut)
            for cnt in reversed(range(estimated_line_cut)):
                if must_break_at_empty_line:
                    if lines[cnt] != "":
                        continue
                print(cnt)
                prev = "\n".join(lines[:cnt])
                post = "\n".join(lines[cnt:])
                if get_token_fn(prev) < limit:
                    break
            if cnt == 0:
                raise RuntimeError("There is an extremely long line of text!")
            # print(len(post))
            result = [prev]
            result.extend(cut(post, must_break_at_empty_line))
            return result
    try:
        return cut(txt, must_break_at_empty_line=True)
    except RuntimeError:
        return cut(txt, must_break_at_empty_line=False)


def force_breakdown(txt, limit, get_token_fn):
    """
    When it is impossible to split with punctuation and blank lines, we use the most violent method to cut
    """
    for i in reversed(range(len(txt))):
        if get_token_fn(txt[:i]) < limit:
            return txt[:i], txt[i:]
    return "Tiktoken unknown error", "Tiktoken unknown error"

def breakdown_txt_to_satisfy_token_limit_for_pdf(txt, get_token_fn, limit):
    # recursion
    def cut(txt_tocut, must_break_at_empty_line, break_anyway=False):  
        if get_token_fn(txt_tocut) <= limit:
            return [txt_tocut]
        else:
            lines = txt_tocut.split('\n')
            estimated_line_cut = limit / get_token_fn(txt_tocut) * len(lines)
            estimated_line_cut = int(estimated_line_cut)
            cnt = 0
            for cnt in reversed(range(estimated_line_cut)):
                if must_break_at_empty_line:
                    if lines[cnt] != "":
                        continue
                prev = "\n".join(lines[:cnt])
                post = "\n".join(lines[cnt:])
                if get_token_fn(prev) < limit:
                    break
            if cnt == 0:
                if break_anyway:
                    prev, post = force_breakdown(txt_tocut, limit, get_token_fn)
                else:
                    raise RuntimeError(f"There is an extremely long line of text!{txt_tocut}")
            # print(len(post))
            # List Recursive 
            result = [prev]
            result.extend(cut(post, must_break_at_empty_line, break_anyway=break_anyway))
            return result
    try:
        # The first attempt, double empty line (\n\n) as the segmentation point
        return cut(txt, must_break_at_empty_line=True)
    except RuntimeError:
        try:
            # The second attempt, using a single blank line (\n) as a split point
            return cut(txt, must_break_at_empty_line=False)
        except RuntimeError:
            try:
                # The third attempt, using the English period (.) as the segmentation point
                res = cut(txt.replace('.', '。\n'), must_break_at_empty_line=False) # 这个中文的句号是故意的，作为一个标识而存在
                return [r.replace('。\n', '.') for r in res]
            except RuntimeError as e:
                try:
                    # The 4th attempt, using the Chinese period (.) as the segmentation point
                    res = cut(txt.replace('。', '。。\n'), must_break_at_empty_line=False)
                    return [r.replace('。。\n', '。') for r in res]
                except RuntimeError as e:
                    # The 5th attempt, there is no way, just cut and perfunctory
                    return cut(txt, must_break_at_empty_line=False, break_anyway=True)


def read_and_clean_pdf_text(fp):
    """
This function is used to split pdf, using a lot of tricks, the logic is messy, and the effect is amazing

     **Input parameter description**
     - `fp`: Path to the pdf file whose text needs to be read and cleaned

     **Output parameter description**
     - `meta_txt`: cleaned text content string
     - `page_one_meta`: List of text content of the first page after cleaning

     **Function function**
     Read the pdf file and clean up the text content. The cleaning rules include:
     - Extract the text information of all block elements and merge them into one string
     - Remove short blocks (less than 100 characters) and replace with carriage returns
     - Clean up extra blank lines
     - Merge paragraph blocks starting with lowercase letters and replace with spaces
     - remove duplicate newlines
     - replaces each newline with two newlines so that each paragraph is separated by two newlines
    """
    import fitz, copy
    import re
    import numpy as np
    # from colorful import print_red,print_green
    fc = 0  # Index 0 text
    fs = 1  # Index 1 font
    fb = 2  # Index 2 frame
    REMOVE_FOOT_NOTE = True # No Discard content that is not the main text (smaller than the main text, such as references, footnotes, legends, etc.)
    REMOVE_FOOT_FFSIZE_PERCENT = 0.95 # If it is less than positive, it is judged not to be the main text (the font size of the main text of some articles is not 100% uniform, and there are small changes that are invisible to the naked eye)? hour
    def primary_ffsize(l):
        """
        Extract text block master font
        """
        fsize_statiscs = {}
        for wtf in l['spans']:
            if wtf['size'] not in fsize_statiscs: fsize_statiscs[wtf['size']] = 0
            fsize_statiscs[wtf['size']] += len(wtf['text'])
        return max(fsize_statiscs, key=fsize_statiscs.get)
        
    def ffsize_same(a,b):
        """
        Extract whether font sizes are approximately equal
        """
        return abs((a-b)/max(a,b)) < 0.02

    with fitz.open(fp) as doc:
        meta_txt = []
        meta_font = []

        meta_line = []
        meta_span = []
        ############################## <Step 1, gather initial information> ##################################
        for index, page in enumerate(doc):
            # file_content += page.get_text()
            text_areas = page.get_text("dict")  # Get the text information on the page
            for t in text_areas['blocks']:
                if 'lines' in t:
                    pf = 998
                    for l in t['lines']:
                        txt_line = "".join([wtf['text'] for wtf in l['spans']])
                        if len(txt_line) == 0: continue
                        pf = primary_ffsize(l)
                        meta_line.append([txt_line, pf, l['bbox'], l])
                        for wtf in l['spans']: # for l in t['lines']:
                            meta_span.append([wtf['text'], wtf['size'], len(wtf['text'])])
                    # meta_line.append(["NEW_BLOCK", pf])
            # block element extraction for each word segment with in line for each line cross-line words                          for each block
            meta_txt.extend([" ".join(["".join([wtf['text'] for wtf in l['spans']]) for l in t['lines']]).replace(
                '- ', '') for t in text_areas['blocks'] if 'lines' in t])
            meta_font.extend([np.mean([np.mean([wtf['size'] for wtf in l['spans']])
                             for l in t['lines']]) for t in text_areas['blocks'] if 'lines' in t])
            if index == 0:
                page_one_meta = [" ".join(["".join([wtf['text'] for wtf in l['spans']]) for l in t['lines']]).replace(
                    '- ', '') for t in text_areas['blocks'] if 'lines' in t]
                
        ############################## <Step 2, get the body main font> ##################################
        fsize_statiscs = {}
        for span in meta_span:
            if span[1] not in fsize_statiscs: fsize_statiscs[span[1]] = 0
            fsize_statiscs[span[1]] += span[2]
        main_fsize = max(fsize_statiscs, key=fsize_statiscs.get)
        if REMOVE_FOOT_NOTE:
            give_up_fize_threshold = main_fsize * REMOVE_FOOT_FFSIZE_PERCENT

        ############################## <Step 3, Split and Reintegrate> ##################################
        mega_sec = []
        sec = []
        for index, line in enumerate(meta_line):
            if index == 0: 
                sec.append(line[fc])
                continue
            if REMOVE_FOOT_NOTE:
                if meta_line[index][fs] <= give_up_fize_threshold:
                    continue
            if ffsize_same(meta_line[index][fs], meta_line[index-1][fs]):
                # try to identify paragraphs
                if meta_line[index][fc].endswith('.') and\
                    (meta_line[index-1][fc] != 'NEW_BLOCK') and \
                    (meta_line[index][fb][2] - meta_line[index][fb][0]) < (meta_line[index-1][fb][2] - meta_line[index-1][fb][0]) * 0.7:
                    sec[-1] += line[fc]
                    sec[-1] += "\n\n"
                else:
                    sec[-1] += " "
                    sec[-1] += line[fc]
            else:
                if (index+1 < len(meta_line)) and \
                    meta_line[index][fs] > main_fsize:
                    # Single line + large font
                    mega_sec.append(copy.deepcopy(sec))
                    sec = []
                    sec.append("# " + line[fc])
                else:
                    # try to identify section
                    if meta_line[index-1][fs] > meta_line[index][fs]:
                        sec.append("\n" + line[fc])
                    else:
                        sec.append(line[fc])
        mega_sec.append(copy.deepcopy(sec))

        finals = []
        for ms in mega_sec:
            final = " ".join(ms)
            final = final.replace('- ', ' ')
            finals.append(final)
        meta_txt = finals

        ############################## <Step 4, messy post-processing> ##################################
        def clear_carraige_return(meta_txt):
            for index, block_txt in enumerate(meta_txt):
                if len(block_txt) < 100:
                    meta_txt[index] = '\n'
            return meta_txt
        meta_txt = clear_carraige_return(meta_txt)

        def clear_extra_blank_lines(meta_txt):
            for index in reversed(range(1, len(meta_txt))):
                if meta_txt[index] == '\n' and meta_txt[index-1] == '\n':
                    meta_txt.pop(index)
            return meta_txt
        meta_txt = clear_extra_blank_lines(meta_txt)

        def merge_para_in_lowercase(meta_txt):
            def starts_with_lowercase_word(s):
                pattern = r"^[a-z]+"
                match = re.match(pattern, s)
                if match:
                    return True
                else:
                    return False
            for _ in range(100):
                for index, block_txt in enumerate(meta_txt):
                    if starts_with_lowercase_word(block_txt):
                        if meta_txt[index-1] != '\n':
                            meta_txt[index-1] += ' '
                        else:
                            meta_txt[index-1] = ''
                        meta_txt[index-1] += meta_txt[index]
                        meta_txt[index] = '\n'
            return meta_txt
        meta_txt = 合并小写开头的段落块(meta_txt)
        meta_txt = 清理多余的空行(meta_txt)

        meta_txt = '\n'.join(meta_txt)
        # remove duplicate newlines
        for _ in range(5):
            meta_txt = meta_txt.replace('\n\n', '\n')

        # newline -> double newline
        meta_txt = meta_txt.replace('\n', '\n\n')

        ############################## <Step 5, show segmentation effect> ##################################
        # for f in finals:
        #    print(f)
        #    print('***************************')

    return meta_txt, page_one_meta
