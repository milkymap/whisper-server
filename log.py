import logging 

logging.basicConfig(
    format='%(asctime)s : (%(name)s) -- %(levelname)7s -- %(message)s',
    level=logging.DEBUG
)

logger = logging.getLogger(name='whisper')