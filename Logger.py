import logging
import os
from datetime import datetime

class Logger :
    def __init__(self):
        self.logger = logging.getLogger('MyLogger')
        self.log_filename = f'log_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'

        if len(self.logger.handlers) == 0:
            # 중복 방지.
            # StreamHandler
            formatter = logging.Formatter(u'%(asctime)s [%(levelname)s] %(message)s')
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)

            self.logger.addHandler(stream_handler)
            self.logger.setLevel(logging.INFO)

        self.log_file()

    def info(self, value):
        self.logger.info(f'{str(value)}')

    def error(self, value):
        self.logger.error(f'{str(value)}')

    def log_file(self):
        file_handler = logging.FileHandler(f'{self.log_filename}')
        file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        self.logger.addHandler(file_handler)
