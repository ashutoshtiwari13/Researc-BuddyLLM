
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

