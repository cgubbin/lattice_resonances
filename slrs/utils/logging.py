import colorlog
import logging
import time
import verboselogs
from tqdm import tqdm

class TqdmHandler(logging.StreamHandler):
    def __init__(self):
        logging.StreamHandler.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)

verboselogs.install()
logger = colorlog.getLogger("slrs")
logger.setLevel(logging.DEBUG)
handler = TqdmHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s [%(levelname)-8s] [%(asctime)s]: %(message)s',
    datefmt='%Y-%d-%d %H:%M:%S',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'light_white',
        'SUCCESS': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white'},))
logger.addHandler(handler)
logger.propagate = False
