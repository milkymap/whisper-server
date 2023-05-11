import click 
import torch as th

import numpy as np

import multiprocessing as mp 
from commands import start_server, start_whisper
from third_party.whisper import load_model, available_models
from third_party.whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE

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
@click.option('--mounting_path', type=str, default='/', help='root_path of the fastapi')

@click.option("--model_name", default="small", type=click.Choice(available_models()), help="name of the Whisper model to use")
@click.option("--model_dir", type=str, default=None, envvar="WHISPER_CACHE", help="the path to save model files; uses ~/.cache/whisper by default")
@click.option("--device", default="cuda" if th.cuda.is_available() else "cpu", help="device to use for PyTorch inference")
@click.option("--verbose/--no-verbose", type=bool, default=True, help="whether to print out the progress and debug messages")
@click.option("--task", default="transcribe", type=click.Choice(["transcribe", "translate"]), help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
@click.option("--language", default=None, type=click.Choice(sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()])), help="language spoken in the audio, specify None to perform language detection")

@click.option("--temperature", type=float, default=0, help="temperature to use for sampling")
@click.option("--best_of", type=int, default=5, help="number of candidates when sampling with non-zero temperature")
@click.option("--beam_size", type=int, default=5, help="number of beams in beam search, only applicable when temperature is zero")
@click.option("--patience", type=float, default=None, help="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search")
@click.option("--length_penalty", type=float, default=None, help="optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default")

@click.option("--suppress_tokens", type=str, default="-1", help="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations")
@click.option("--initial_prompt", type=str, default=None, help="optional text to provide as a prompt for the first window.")
@click.option("--condition_on_previous_text/--no-condition_on_previous_text", default=True, help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop")
@click.option("--fp16/--no-fp16", default=True, help="whether to perform inference in fp16; True by default")

@click.option("--temperature_increment_on_fallback", type=float, default=0.2, help="temperature to increase when falling back when the decoding fails to meet either of the thresholds below")
@click.option("--compression_ratio_threshold", type=float, default=2.4, help="if the gzip compression ratio is higher than this value, treat the decoding as failed")
@click.option("--logprob_threshold", type=float, default=-1.0, help="if the average log probability is lower than this value, treat the decoding as failed")
@click.option("--no_speech_threshold", type=float, default=0.6, help="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence")
@click.option("--threads", type=int, default=0, help="number of threads used by torch for CPU inference; supercedes MKL_NUM_THREADS/OMP_NUM_THREADS")

@click.pass_context
def start_service(
    ctx:click.core.Context, 
    port:int, 
    host:str, 
    mounting_path:str,
    
    model_name, 
    model_dir, 
    device, 
    verbose, 
    task, 
    language, 

    temperature, 
    best_of, 
    beam_size,
    patience, 
    length_penalty,

    suppress_tokens, 
    initial_prompt, 
    condition_on_previous_text,
    fp16, 

    temperature_increment_on_fallback, 
    compression_ratio_threshold, 
    logprob_threshold, 
    no_speech_threshold, 
    threads):


    params_map = ctx.params
    for key,val in params_map.items():
        logger.debug(f'{key:<25} : {val}')

    if temperature_increment_on_fallback is not None:
        params_map['temperature'] = tuple(np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback))
    else:
        params_map['temperature'] = [temperature]

    if threads > 0:
        th.set_num_threads(threads)


    processes:List[mp.Process] = [
        mp.Process(target=start_server, kwargs={'port':port, 'host': host, 'workdir': ctx.obj['workdir'], 'mounting_path': mounting_path}),
        mp.Process(target=start_whisper, kwargs={'params_map': params_map})
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
    th.multiprocessing.set_start_method('spawn')
    group(obj={})
