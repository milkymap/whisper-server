import zmq 
import json 

import operator as op

from api_schema import ZeroMQConfig, TranscriptionStatus
from log import logger

from os import path 
from io import StringIO
from typing import List, Dict 

import subprocess
from third_party.whisper import load_model
from third_party.whisper.transcribe import transcribe 



from typing import Dict, Any

class ZMQWhisper:
    def __init__(self, params_map:Dict[str, Any]):
        self.params_map = params_map

    def format_timestamp(self, seconds: float, always_include_hours: bool = False, decimal_marker: str = '.'):
        assert seconds >= 0, "non-negative timestamp expected"
        milliseconds = round(seconds * 1000.0)

        hours = milliseconds // 3_600_000
        milliseconds -= hours * 3_600_000

        minutes = milliseconds // 60_000
        milliseconds -= minutes * 60_000

        seconds = milliseconds // 1_000
        milliseconds -= seconds * 1_000

        hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
        return f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"

    def to_srt(self, segments: List[Dict]):    
        acc = []
        for item in segments:
            id, start, end, text = op.itemgetter('id', 'start', 'end', 'text')(item)
            acc.append(
                f"{id+1}\n{self.format_timestamp(start, always_include_hours=True, decimal_marker=',')} --> {self.format_timestamp(end, always_include_hours=True, decimal_marker=',')}\n{text.strip().replace('>', '->')}\n"
            )
        
        return '\n'.join(acc)
    
    def to_video(self, path2audio_file, path2blank_video_file, path2subtitled_file, path2subtitled_video_file):
        logger.debug('video generation has started ...!')
       
        subprocess.run(
            [
                'ffmpeg', 
                '-y',
                '-stream_loop', '1',
                '-f',
                'lavfi', 
                '-i', 
                'color=size=640x480:rate=25:color=black', 
                '-i', 
                path2audio_file, 
                '-shortest', 
                '-map', '0:v', 
                '-map', '1:a', 
                '-c:v', 'libx264', 
                '-c:a', 'copy', 
                path2blank_video_file
            ]
        )
        
        subprocess.run(
            ['ffmpeg', '-y' ,'-i', path2blank_video_file, '-vf', f'subtitles={path2subtitled_file}', path2subtitled_video_file]
        )

    def run(self):

        logger.debug('whisper server is running : ..!')

        keep_loop = True 
        while keep_loop:
            try:
                socket_status = self.router_socket.poll(1000)
                if socket_status == zmq.POLLIN:
                    client_id, _, encoded_message = self.router_socket.recv_multipart()
                    path2audio = encoded_message.decode()
                    self.router_socket.send_multipart([client_id, b''], flags=zmq.SNDMORE)
                    self.router_socket.send_string(TranscriptionStatus.RUNNING)
                    
                    try:
                        transcription = transcribe(self.model, path2audio, **self.params_map)

                        basepath, audiofile = path.split(path2audio)
                        filename, _ = audiofile.split('.')
                        
                        path2text = path.join(basepath, 'data.txt')
                        path2segments = path.join(basepath, 'segments.srt')
                        path2blank_video_file= path.join(basepath, 'blank.mp4')
                        path2subtitled_video_file= path.join(basepath, 'subtitled.mp4')

                        path2response_json = path.join(basepath, f'transcription.json')

                        with open(path2response_json, mode='w') as fp:
                            json.dump({
                                    'text': transcription['text'],
                                    'srt_segments': transcription['segments']
                                },
                                fp
                            )
                        
                        with open(path2text, mode='w') as fp:
                            fp.write(transcription['text'])
                        
                        with open(path2segments, mode='w') as fp:
                            fp.write(self.to_srt(transcription['segments']))
                        
                        self.to_video(
                            path2audio_file=path2audio,
                            path2blank_video_file=path2blank_video_file,
                            path2subtitled_file=path2segments,
                            path2subtitled_video_file=path2subtitled_video_file
                        )
                        
                        self.router_socket.send_multipart([client_id, b''], flags=zmq.SNDMORE)
                        self.router_socket.send_string(TranscriptionStatus.COMPLETED)
                    except Exception as e:
                        self.router_socket.send_multipart([client_id, b''], flags=zmq.SNDMORE)
                        self.router_socket.send_string(TranscriptionStatus.FAILED)
                        logger.exception(e)
            except KeyboardInterrupt:
                keep_loop = False 
            except Exception as e:
                logger.exception(e)
        # end while loop ...! 

        self.publisher_socket.send_multipart([b'TERMINATE', b''])


    def __enter__(self):
        self.model = load_model(
            self.params_map['model_name'], 
            device=self.params_map['device'], 
            download_root=self.params_map['model_dir']
        )


        self.params_map.pop('model_name')
        self.params_map.pop('model_dir')
        self.params_map.pop('threads')
        self.params_map.pop('device')
        
        self.params_map.pop('port')
        self.params_map.pop('mounting_path')
        self.params_map.pop('host')
        self.params_map.pop("temperature_increment_on_fallback")

        logger.info('whisper model was loaded')

        self.ctx = zmq.Context()
        self.router_socket:zmq.Socket = self.ctx.socket(zmq.ROUTER)
        self.publisher_socket:zmq.Socket = self.ctx.socket(zmq.PUB)

        self.router_socket.bind(ZeroMQConfig.ROUTER_ADDRESS)
        self.publisher_socket.bind(ZeroMQConfig.PUBLISHER_ADDRESS)
        logger.debug('whisper has initialized all ressources')
        return self 
    
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            logger.warning(f'Exception => {exc_value}')
            logger.exception(traceback)
        self.router_socket.close(linger=0)
        self.publisher_socket.close(linger=0)
        logger.debug('whisper has released all ressources')
