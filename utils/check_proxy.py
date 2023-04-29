def check_proxy(proxies):
    import requests
    proxies_https = proxies['https'] if proxies is not None else '无'
    try:
        response = requests.get("https://ipapi.co/json/",
                                proxies=proxies, timeout=4)
        data = response.json()
        print(f'Query the geographical location of the agent, and the returned result is {data}')
        if 'country_name' in data:
            country = data['country_name']
            result = f"Proxy configuration {proxies_https}, proxy location: {country}"
        elif 'error' in data:
            result = f"Proxy configuration {proxies_https}, proxy location: unknown, IP query frequency limited"
        print(result)
        return result
    except:
        result = f"Proxy configuration {proxies_https}, proxy location query timed out, proxy may be invalid"
        print(result)
        return result


def backup_and_download(current_version, remote_version):
    """
    One-click update protocol: backup and download
    """
    from functional_utils import get_conf
    import shutil
    import os
    import requests
    import zipfile
    os.makedirs(f'./history', exist_ok=True)
    backup_dir = f'./history/backup-{current_version}/'
    new_version_dir = f'./history/new-version-{remote_version}/'
    if os.path.exists(new_version_dir):
        return new_version_dir
    os.makedirs(new_version_dir)
    shutil.copytree('./', backup_dir, ignore=lambda x, y: ['history'])
    proxies, = get_conf('proxies')
    r = requests.get(
        'https://github.com/ashutoshtiwari13/Researc-BuddyLLM/zip/refs/heads/master.zip', proxies=proxies, stream=True)
    zip_file_path = backup_dir+'/master.zip'
    with open(zip_file_path, 'wb+') as f:
        f.write(r.content)
    dst_path = new_version_dir
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        for zip_info in zip_ref.infolist():
            dst_file_path = os.path.join(dst_path, zip_info.filename)
            if os.path.exists(dst_file_path):
                os.remove(dst_file_path)
            zip_ref.extract(zip_info, dst_path)
    return new_version_dir


def patch_and_restart(path):
    """
    One-click update protocol: overwrite and reboot
    """
    from distutils import dir_util
    import shutil
    import os
    import sys
    import time
    import glob
    # if not using config_private, move origin config.py as config_private.py
    if not os.path.exists('config_private.py'):
        print('Since you have not set config_private.py private configuration, now move your existing configuration to config_private.py to prevent configuration loss,',
               'In addition, you can retrieve the old version of the program in the history subfolder at any time.')
        shutil.copyfile('config.py', 'config_private.py')
    path_new_version = glob.glob(path + '/*-master')[0]
    dir_util.copy_tree(path_new_version, './')
    print('The code has been updated, and the pip package dependency will be updated soon...')
    for i in reversed(range(5)): time.sleep(1); print(i)
    try: 
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
    except:
        print亮红('There is a problem with the pip package dependency installation, you need to manually install the new dependency library `python -m pip install -r requirements.txt`, and then use the regular `python main.py` to start')
    print亮绿('After the update is complete, you can retrieve the old version of the program in the history subfolder at any time, and restart after 5s')
    print亮红('If the restart fails, you may need to manually install the newly added dependency library `python -m pip install -r requirements.txt`, and then start it in the normal `python main.py` way.')
    print(' ------------------------------ -----------------------------------')
    for i in reversed(range(8)): time.sleep(1); print(i)
    os.execl(sys.executable, sys.executable, *sys.argv)


def get_current_version():
    import json
    try:
        with open('./version', 'r', encoding='utf8') as f:
            current_version = json.loads(f.read())['version']
    except:
        current_version = ""
    return current_version


def auto_update():
    """
    One-click update agreement: query version and user opinions
    """
    try:
        from toolbox import get_conf
        import requests
        import time
        import json
        proxies, = get_conf('proxies')
        response = requests.get(
            "https://raw.githubusercontent.com/ashutoshtiwari13/Researc-BuddyLLM/master/version", proxies=proxies, timeout=5)
        remote_json_data = json.loads(response.text)
        remote_version = remote_json_data['version']
        if remote_json_data["show_feature"]:
            new_feature = "new function:" + remote_json_data["new_feature"]
        else:
            new_feature = ""
        with open('./version', 'r', encoding='utf8') as f:
            current_version = f.read()
            current_version = json.loads(current_version)['version']
        if (remote_version - current_version) >= 0.01:
            from colorful import print亮黄
            print亮黄(
                f'\nA new version is available. New version: {remote_version}, current version: {current_version}. {new_feature}')
            print('（1）Github update address:\nhttps://github.com/ashutoshtiwari13/Researc-BuddyLLM\n')
            user_instruction = input('(2) Is it possible to update the code with one click (Y+Enter=confirm, input other/no input+Enter=no update)?')
            if user_instruction in ['Y', 'y']:
                path = backup_and_download(current_version, remote_version)
                try:
                    patch_and_restart(path)
                except:
                    print('Update failed.')
            else:
                print('Auto Updater: Disabled')
                return
        else:
            return
    except:
        print('Auto Updater: Disabled')

def warm_up_modules():
    print('Performing warmup of some modules...')
    from core_llm.bridge_all_llm import model_info
    enc = model_info["gpt-3.5-turbo"]['tokenizer']
    enc.encode("Module preheating", disallowed_special=())
    enc = model_info["gpt-4"]['tokenizer']
    enc.encode("Module preheating", disallowed_special=())

if __name__ == '__main__':
    import os
    os.environ['no_proxy'] = '*'  # Avoid accidental pollution of the proxy network
    from toolbox import get_conf
    proxies, = get_conf('proxies')
    check_proxy(proxies)
