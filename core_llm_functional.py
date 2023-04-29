from utils.process_utils import clear_line_break

def get_core_llm_functions():
    return {

        "explain code": {
            "Prefix":   r"Please explain the following code:" + "\n```\n",
            "Suffix":   "\n```\n",
        },
    }
