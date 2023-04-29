"""
Functions in this file :

get_token_usage_data:
write_results_to_file:
regular_txt_to_markdown:
report_exception:
text_to_HTML_breaks:
markdown_to_HTML:
chatbot_reply_completion_marker:
format_io:
extract_archive
"""


def get_token_usage_data(text):
    try:
        text = "maximum context length is 4097 tokens, Your message resulted in exceed of token"
        pattern = r"(\d+)\s+tokens\b"
        match = re.findall(pattern, text)
        EXCEED_ALLOWED = 500
        max_limit = float(match[0]) - EXCEED_ALLOWED
        current_tokens = float(match[1])
        ratio = max_limit/current_tokens
        assert ratio > 0 and ratio < 1
        return ratio, str(int(current_tokens-max_limit))
    except:
        return 0.5, 'Unknown'


def write_results_to_file(history, file_name=None):
    """
    Writes the conversation history in a MarDown(.md) file.
    If No filename provided - uses the current datetimestamp to create the file
    """

    import os
    import time
    if file_name is None:
        file_name = 'Coversation_Report' +
                           time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '.md'
    os.makedirs('./conversation_log/',exist_ok =True)

    with open(f'./conversation_log/{file_name}', 'w', encoding='utf8') as f:
        f.write('CONVERSATION REPORT\n')
        for i, content in enumerate(history):
            try:
                if type(content) != str:
                    content = str(content)
            except:
                continue
            if i % 2 == 0:
                f.write('## ')
            f.write(content)
            f.write('\n\n')
    res = str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())) + 'Conversation Recorded in the file' + os.path.abspath(f'./conversation_log/{file_name}')
    print(res)
    return res

def regular_txt_to_markdown(text):
    """
    Converts Regular text to Markdown
    """
    text = text.replace('\n', '\n\n')
    text = text.replace('\n\n\n', '\n\n')
    text = text.replace('\n\n\n', '\n\n')
    return text


def report_exception(chatbot, history, a, b):
    """
    Add Conversation errors
    """
    chatbot.append((a, b))
    history.append(a)
    history.append(b)

def text_to_HTML_breaks(text):
    if '```' in text:
        return text
    else:
        lines = text.split("\n")
        for i, line in enumerate(lines):
            lines[i] = ines[i].replace( "", "&nbsp;")
        text = "</br>".join(lines)
        return text    


def markdown_to_HTML(text):
    """
    Converts a Markdown file to an HTML format 
    """

    pre = '<div class="markdown-body">'
    suf = '</div>'
    markdown_extension_configs = {
        'mdx_math': {
            'enable_dollar_delimiter': True,
            'use_gitlab_delimiters': False,
        },
    }

    find_equation_pattern = r'<script type="math/tex(?:.*?)>(.*?)</script>'

    def tex2mathml_catch_exception(content, *args, **kwargs):
        try:
            content = tex2mathml(content, *args, **kwargs)
        except:
            content = content
        return content

    def replace_math_no_render(match):
        content = match.group(1)
        if 'mode=display' in match.group(0):
            content = content.replace('\n', '</br>')
            return f"<font color=\"#00FF00\">$$</font><font color=\"#FF00FF\">{content}</font><font color=\"#00FF00\">$$</font>"
        else:
            return f"<font color=\"#00FF00\">$</font><font color=\"#FF00FF\">{content}</font><font color=\"#00FF00\">$</font>"

    def replace_math_render(match):
        content = match.group(1)
        if 'mode=display' in match.group(0):
            if '\\begin{aligned}' in content:
                content = content.replace('\\begin{aligned}', '\\begin{array}')
                content = content.replace('\\end{aligned}', '\\end{array}')
                content = content.replace('&', ' ')
            content = tex2mathml_catch_exception(content, display="block")
            return content
        else:
            return tex2mathml_catch_exception(content)

    def markdown_bug_hunt(content):
        content = content.replace('<script type="math/tex">\n<script type="math/tex; mode=display">', '<script type="math/tex; mode=display">')
        content = content.replace('</script>\n</script>', '</script>')
        return content   

    if ('$' in txt) and ('```' not in txt):  
        # convert everything to html format
        split = markdown.markdown(text='---')
        convert_stage_1 = markdown.markdown(text=txt, extensions=['mdx_math', 'fenced_code', 'tables', 'sane_lists'], extension_configs=markdown_extension_configs)
        convert_stage_1 = markdown_bug_hunt(convert_stage_1)
        # re.DOTALL: Make the '.' special character match any character at all, including a newline; without this flag, '.' will match anything except a newline. Corresponds to the inline flag (?s).
        # 1. convert to easy-to-copy tex (do not render math)
        convert_stage_2_1, n = re.subn(find_equation_pattern, replace_math_no_render, convert_stage_1, flags=re.DOTALL)
        # 2. convert to rendered equation
        convert_stage_2_2, n = re.subn(find_equation_pattern, replace_math_render, convert_stage_1, flags=re.DOTALL)
        # cat them together
        return pre + convert_stage_2_1 + f'{split}' + convert_stage_2_2 + suf
    else:
        return pre + markdown.markdown(txt, extensions=['fenced_code', 'codehilite', 'tables', 'sane_lists']) + suf         


def chatbot_reply_completion_marker(chatbot_reply):

    if '```' not in chatbot_reply:
        return chatbot_reply
    if chatbot_reply.endswith('```'):
        return chatbot_reply

    segments = chatbot_reply.split('```')
    n_mark = len(segments) - 1
    if n_mark % 2 == 1:
        print('Output Code Segment')
        return chatbot_reply+'\n```'
    else:
        return chatbot_reply

def format_io(self, y):
    """
    Parse input and output in HTML formats
    """
    if y is None or y == []:
        return []
    i_ask, chatbot_reply = y[-1]
    i_ask = text_to_HTML_breaks(i_ask)
    chatbot_reply = chatbot_reply_completion_marker(chatbot_reply) 
    y[-1] = (
        None if i_ask is None else markdown.markdown(i_ask, extensions=['fenced_code', 'tables']),
        None if chatbot_reply is None else markdown_convertion(chatbot_reply)
    )
    return y

def extract_archive(file_path, dest_dir):
    """
    Code from Stack Overflow
    """
    import zipfile
    import tarfile
    import os
    # Get the file extension of the input file
    file_extension = os.path.splitext(file_path)[1]

    # Extract the archive based on its extension
    if file_extension == '.zip':
        with zipfile.ZipFile(file_path, 'r') as zipobj:
            zipobj.extractall(path=dest_dir)
            print("Successfully extracted zip archive to {}".format(dest_dir))

    elif file_extension in ['.tar', '.gz', '.bz2']:
        with tarfile.open(file_path, 'r:*') as tarobj:
            tarobj.extractall(path=dest_dir)
            print("Successfully extracted tar archive to {}".format(dest_dir))

    elif file_extension == '.rar':
        try:
            import rarfile
            with rarfile.RarFile(file_path) as rf:
                rf.extractall(path=dest_dir)
                print("Successfully extracted rar archive to {}".format(dest_dir))
        except:
            print("Rar format requires additional dependencies to install")
            return '\n\nNeed to Install pip install rarfile'

    # Need to execute - pip install py7zr
    elif file_extension == '.7z':
        try:
            import py7zr
            with py7zr.SevenZipFile(file_path, mode='r') as f:
                f.extractall(path=dest_dir)
                print("Successfully extracted 7z archive to {}".format(dest_dir))
        except:
            print("7z format requires additional dependencies to install")
            return '\n\nNeed to execute pip install py7zr'
    else:
        return ''
    return ''


def find_recent_files(directory):
    """
        me: find files that is created with in one minutes under a directory with python, write a function
        gpt: here it is!
    """
    import os
    import time
    current_time = time.time()
    one_minute_ago = current_time - 60
    recent_files = []

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if file_path.endswith('.log'):
            continue
        created_time = os.path.getmtime(file_path)
        if created_time >= one_minute_ago:
            if os.path.isdir(file_path):
                continue
            recent_files.append(file_path)

    return recent_files


def on_file_uploaded(files, chatbot, txt, txt2, checkboxes):
    """
    Callback function on file is uploaded
    """
    if len(files) == 0:
        return chatbot, txt
    import shutil
    import os
    import time
    import glob
    from process_utils import extract_archive
    try:
        shutil.rmtree('./private_upload/')
    except:
        pass
    time_tag = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    os.makedirs(f'private_upload/{time_tag}', exist_ok=True)
    err_msg = ''
    for file in files:
        file_origin_name = os.path.basename(file.orig_name)
        shutil.copy(file.name, f'private_upload/{time_tag}/{file_origin_name}')
        err_msg += extract_archive(f'private_upload/{time_tag}/{file_origin_name}',
                                   dest_dir=f'private_upload/{time_tag}/{file_origin_name}.extract')
    moved_files = [fp for fp in glob.glob('private_upload/**/*', recursive=True)]
    if "checked" in checkboxes:
        txt = ""
        txt2 = f'private_upload/{time_tag}'
    else:
        txt = f'private_upload/{time_tag}'
        txt2 = ""
    moved_files_str = '\t\n\n'.join(moved_files)
    chatbot.append(['File Upload Complete',
                    f'[Local Message] Recieved the Following files: \n\n{moved_files_str}' +
                    f'\n\nCall path parameter has been corrected to: \n\n{txt}' +
                    f'\n\nClick on the red link to to use it as a input file'+err_msg])
    return chatbot, txt, txt2


def on_report_generated(files, chatbot):
    from toolbox import find_recent_files
    report_files = find_recent_files('gpt_log')
    if len(report_files) == 0:
        return None, chatbot
    # files.extend(report_files)
    chatbot.append(['Remote Summary Reports', 'Summary Report added','Please check'])
    return report_files, chatbot

def clear_line_break(txt):
    txt = txt.replace('\n', ' ')
    txt = txt.replace('  ', ' ')
    txt = txt.replace('  ', ' ')
    return txt

def clip_history(inputs, history, tokenizer, max_token_limit):
    """
    reduce the length of history by clipping.
    this function search for the longest entries to clip, little by little,
    until the number of token of history is reduced under threshold.
    """
    import numpy as np
    from request_llm.bridge_all_llm import model_info
    def get_token_num(txt): 
        return len(tokenizer.encode(txt, disallowed_special=()))
    input_token_num = get_token_num(inputs)
    if input_token_num < max_token_limit * 3 / 4:
        # When the proportion of token in the input part is less than 3/4 of the limit, when cropping
        # 1. Leave the margin for input
        max_token_limit = max_token_limit - input_token_num
        # 2. Leave margin for output
        max_token_limit = max_token_limit - 128
        # 3. If the margin is too small, clear the history directly
        if max_token_limit < 128:
            history = []
            return history
    else:
        # When the proportion of tokens in the input part > 3/4 of the limit, clear the history directly
        history = []
        return history

    everything = ['']
    everything.extend(history)
    n_token = get_token_num('\n'.join(everything))
    everything_token = [get_token_num(e) for e in everything]

    # Granularity at truncation
    delta = max(everything_token) // 16

    while n_token > max_token_limit:
        where = np.argmax(everything_token)
        encoded = tokenizer.encode(everything[where], disallowed_special=())
        clipped_encoded = encoded[:len(encoded)-delta]
        everything[where] = tokenizer.decode(clipped_encoded)[:-1]    # -1 to remove the may-be illegal char
        everything_token[where] = get_token_num(everything[where])
        n_token = get_token_num('\n'.join(everything))

    history = everything[1:]
    return history

class DummyWith():
    """
    This code defines an empty context manager called DummyWith,
     A context manager is a Python object intended for use with the with statement,
     to ensure that some resources are properly initialized and cleaned up during code block execution.
     A context manager must implement two methods, __enter__() and __exit__().
     In the case where context execution starts, the __enter__() method is called before the code block is executed,
     At the end of context execution, the __exit__() method will be called.
    """
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return
