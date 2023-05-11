
from typing import Dict, Any
from server import APIServer
from worker import ZMQWhisper

def start_server(port:int, host:str, workdir:str, mounting_path:str):
    with APIServer(port=port, host=host, workdir=workdir, mounting_path=mounting_path) as api_server:
        api_server.run()

def start_whisper(params_map:Dict[str, Any]):
    with ZMQWhisper(params_map=params_map) as model:
        model.run()
