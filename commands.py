
from server import APIServer
from worker import ZMQWhisper

def start_server(port:int, host:str, workdir:str):
    with APIServer(port=port, host=host, workdir=workdir) as api_server:
        api_server.run()

def start_whisper(model_size:str, path2whisper_cache:str):
    with ZMQWhisper(model_size=model_size, path2whisper_cache=path2whisper_cache) as model:
        model.run()
