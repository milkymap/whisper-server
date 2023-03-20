from enum import Enum 
from pydantic import BaseModel

class ZeroMQConfig(str, Enum):
    ROUTER_ADDRESS:str='ipc://router_address.ipc'
    PUBLISHER_ADDRESS:str='ipc://publisher_address.ipc'

class TranscriptionStatus(str, Enum):
    FAILED:str='failed'
    PENDING:str='pending'
    RUNNING:str='running'
    TIMEOUT:str='timeout'
    COMPLETED:str='completed'
    UNDEFINED:str='undefined'
    INTERRUPTED:str='interrupted'

class TranscriptionRequest(BaseModel):
    dirname:str

class TranscriptionResponse(BaseModel):
    task_id:str 

class MonitoringResponse(BaseModel):
    task_id:str 
    task_status:TranscriptionStatus