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

def run_gradio_in_subpath(demo, auth, port, custom_path):
    """
    Change the running address of gradio to the specified secondary path
    """
    def is_path_legal(path: str)->bool:
        '''
        check path for sub url
        path: path to check
        return value: do sub url wrap
        '''
        if path == "/": return True
        if len(path) == 0:
            print("ilegal custom path: {}\npath must not be empty\ndeploy on root url".format(path))
            return False
        if path[0] == '/':
            if path[1] != '/':
                print("deploy on sub-path {}".format(path))
                return True
            return False
        print("ilegal custom path: {}\npath should begin with \'/\'\ndeploy on root url".format(path))
        return False

    if not is_path_legal(custom_path): raise RuntimeError('Ilegal custom path')
    import uvicorn
    import gradio as gr
    from fastapi import FastAPI
    app = FastAPI()
    if custom_path != "/":
        @app.get("/")
        def read_main(): 
            return {"message": f"Gradio is running at: {custom_path}"}
    app = gr.mount_gradio_app(app, demo, path=custom_path)
    uvicorn.run(app, host="0.0.0.0", port=port) # , auth=auth