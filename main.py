import click 

import multiprocessing as mp 
from commands import start_server, start_whisper

from typing import List 
from log import logger 

@click.group(chain=False, invoke_without_command=True)
@click.option('--use_gpu/--no-use_gpu', type=bool)
@click.option('--workdir', type=click.Path(exists=True), envvar='WORKDIR', required=True)
@click.option('--path2whisper_cache', envvar='WHISPER_CACHE', type=click.Path(exists=True), required=True)
@click.pass_context
def group(ctx:click.core.Context, use_gpu:bool, workdir:str, path2whisper_cache:str):
    ctx.ensure_object(dict)
    ctx.obj['use_gpu'] = use_gpu
    ctx.obj['workdir'] = workdir
    ctx.obj['path2whisper_cache'] = path2whisper_cache

@group.command()
@click.option('--port', type=int, default=8000)
@click.option('--host', type=str, default='0.0.0.0')
@click.option('--model_size', type=click.Choice(['tiny', 'base', 'small', 'medium', 'large']), default='base')
@click.pass_context
def start_service(ctx:click.core.Context, port:int, host:str, model_size:str):
    processes:List[mp.Process] = [
        mp.Process(target=start_server, kwargs={'port':port, 'host': host, 'workdir': ctx.obj['workdir']}),
        mp.Process(target=start_whisper, kwargs={'model_size': model_size, 'path2whisper_cache': ctx.obj['path2whisper_cache']})
    ]

    for process_ in processes:
        process_.start()
    
    keep_loop = True 
    while keep_loop:
        try:
            process_states = [ process_.is_alive() for process_ in processes ]
            keep_loop = all(process_states)
        except KeyboardInterrupt:
            keep_loop = False 
        except Exception as e:
            logger.error(e)

    for process_ in processes:
        process_.terminate()

    for process_ in processes:
        process_.join()

if __name__ == '__main__':
    group(obj={})
