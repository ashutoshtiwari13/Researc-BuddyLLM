API_KEY = "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" #main engine API key for Large langauge model

USE_PROXY = False
if USE_PROXY:
    proxies = {
        "http":  "socks5h://localhost:11284",
        "https": "socks5h://localhost:11284",
    }
else:
    proxies = None

DEFAULT_WORKER_NUM = 3

CHATBOT_HEIGHT = 1115

LAYOUT = "LEFT-RIGHT" 
DARK_MODE = True 

TIMEOUT_SECONDS = 30

WEB_PORT = -1

MAX_RETRY = 2

LLM_MODEL = "gpt-3.5-turbo"
AVAIL_LLM_MODELS = ["gpt-3.5-turbo", "gpt-4"]


LOCAL_MODEL_DEVICE = "cpu" 

CONCURRENT_COUNT = 100

AUTHENTICATION = []

API_URL_REDIRECT = {}

CUSTOM_PATH = "/"

NEWBING_STYLE = "creative"  # ["creative", "balanced", "precise"]
NEWBING_COOKIES = """
your bing cookies here
"""
