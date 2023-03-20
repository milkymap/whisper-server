import zmq 
import zmq.asyncio as zio 

import uvicorn
import pickle 
import aiofile
from uuid import uuid4

import signal 
import asyncio

from time import time 
from typing import List, Dict, Tuple, Optional, Any
from os import path 

from log import logger 

from asyncio import Lock, Event, Semaphore, Condition
from fastapi import FastAPI, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse

from api_schema import ZeroMQConfig, TranscriptionStatus, TranscriptionResponse, MonitoringResponse, TranscriptionRequest

class APIServer:
    def __init__(self, port:int, host:str, workdir:str):
        self.port = port 
        self.host = host 
        self.workdir = workdir 

        self.path2memories = path.join(self.workdir, 'memories.pkl')
        self.map_pipeline2status:Dict[str, TranscriptionStatus] = {}
    
    async def handle_startup(self):
        if path.isfile(self.path2memories):
            async with aiofile.async_open(self.path2memories, mode='rb') as fp:
                binarystream = await fp.read()
                self.map_pipeline2status = pickle.loads(binarystream)
                logger.info('memories was loaded from previous snapshot')

        self.ctx = zio.Context()
        self.mutex = Lock()
        self.liveness = Event()
        self.access_card = Semaphore(value=2048)

        self.ctx.set(zmq.MAX_SOCKETS, 2048)

        self.loop = asyncio.get_running_loop()
        self.loop.add_signal_handler(
            sig=signal.SIGINT,
            callback=lambda: self.liveness.clear()  # stop all background task 
        )

        self.liveness.set()
        
    async def handle_shutdown(self):
        logger.debug(f'server liveness : {self.liveness.is_set()}')
        self.ctx.term() 
        async with aiofile.async_open(self.path2memories, mode='wb') as fp:
            binarystream = pickle.dumps(self.map_pipeline2status)
            await fp.write(binarystream)
            logger.info('memories was saved ...!')
    
    async def handle_monitoring(self, task_id:str):
        async with self.mutex:
            task_status = self.map_pipeline2status.get(task_id, None)
            if task_status is None:
                return JSONResponse(
                    status_code=500,
                    content=MonitoringResponse(
                        task_id=task_id,
                        task_status=TranscriptionStatus.UNDEFINED
                    ).dict()
                )

            return JSONResponse(
                status_code=200,
                content=MonitoringResponse(
                    task_id=task_id,
                    task_status=task_status
                ).dict()
            )

    async def handle_background_transcription(self, task_id:str, path2audio:str):
        async with self.mutex:
            self.map_pipeline2status[task_id] = TranscriptionStatus.PENDING

        async with self.access_card:
            timeout = 300  # 5mn 
            dealer_socket:zio.Socket = self.ctx.socket(zmq.DEALER)
            subscriber_socket:zio.Socket = self.ctx.socket(zmq.SUB)
            subscriber_socket.setsockopt_string(zmq.SUBSCRIBE, 'TERMINATE')
            
            dealer_socket.connect(ZeroMQConfig.ROUTER_ADDRESS)
            subscriber_socket.connect(ZeroMQConfig.PUBLISHER_ADDRESS)
            
            poller = zio.Poller()
            poller.register(dealer_socket, zmq.POLLIN)
            poller.register(subscriber_socket, zmq.POLLIN)
            
            await dealer_socket.send_multipart([b'', path2audio.encode()])
            logger.debug(f"{task_id} was schedulled")
            
            try:
                start = None 
                keep_loop = True 
                timer_is_set = False 
                while keep_loop and self.liveness.is_set():  
                    if timer_is_set and start is not None:
                        end = time()
                        duration = end - start 
                        if duration > timeout:  # 2mn
                            raise TimeoutError('transcription has taken to much time ... (2mn)')
                    map_socket2status = dict(await poller.poll(timeout=100))
                    if dealer_socket in map_socket2status:
                        if map_socket2status[dealer_socket] == zmq.POLLIN:
                            _, encoded_message = await dealer_socket.recv_multipart()
                            message:TranscriptionStatus = encoded_message.decode()            
                            async with self.mutex:
                                self.map_pipeline2status[task_id] = message  # RUNNING | FAILED | COMPLETED
                                if message in [TranscriptionStatus.FAILED, TranscriptionStatus.COMPLETED]:
                                    logger.debug(f"{task_id} has finished")
                                    keep_loop = False 
                                else:
                                    logger.debug(f"{task_id} is running on the background(whisper transcription)")
                                    timer_is_set = True 
                                    start = time()
                              
                    if subscriber_socket in map_socket2status:
                        if map_socket2status[subscriber_socket] == zmq.POLLIN:
                            topic, _ = await subscriber_socket.recv_multipart()
                            if topic == b'TERMINATE':
                                keep_loop = False 
                                async with self.mutex:
                                    self.map_pipeline2status[task_id] = TranscriptionStatus.INTERRUPTED
                                    logger.warning(f"{task_id} was interrupted")
                # end while loop ...! 
            except TimeoutError:
                # end while loop monitoring 
                logger.warning(f"{task_id} timeout...!")
                async with self.mutex:
                    self.map_pipeline2status[task_id] = TranscriptionStatus.TIMEOUT
            except Exception as e:
                logger.exception(e)

            poller.unregister(dealer_socket)
            poller.unregister(subscriber_socket)

            dealer_socket.close(linger=0)
            subscriber_socket.close(linger=0)
            logger.debug(f'{task_id} was closed ...!')
        # end acces card context manager (semaphore)

    async def handle_transcription(self, background_tasks:BackgroundTasks, incoming_req:TranscriptionRequest):
        try:
            dirname = incoming_req.dirname
           
            path2audio = path.join(self.workdir, dirname, 'source.mp3')
           
            if path.isfile(path2audio):
                task_id = str(uuid4())
                background_tasks.add_task(
                    self.handle_background_transcription,
                    task_id,
                    path2audio
                )

                logger.debug(f'server has scheduled the transciption of the file {path2audio}')

                return JSONResponse(
                    status_code=200,
                    content=TranscriptionResponse(
                        task_id=task_id
                    ).dict()
                )
            raise FileNotFoundError(f'{path2audio} is not a valid file')
        except Exception as e:
            error_message = f'Exception => {e}' 
            logger.exception(error_message)
            return JSONResponse(
                status_code=500,
                content=error_message
            )


    def run(self):
        uvicorn.run(app=self.core, port=self.port, host=self.host)

    def __enter__(self):
        self.core = FastAPI()
        
        self.core.add_event_handler(event_type='startup', func=self.handle_startup)
        self.core.add_event_handler(event_type='shutdown', func=self.handle_shutdown)
        
        self.core.add_api_route(path='/monitoring', endpoint=self.handle_monitoring, methods=['GET'], response_model=MonitoringResponse)
        self.core.add_api_route(path='/transcription', endpoint=self.handle_transcription, methods=['POST'], response_model=TranscriptionResponse)

        return self 
    
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            logger.warning(f'Exception => {exc_value}')
            logger.exception(traceback)
        