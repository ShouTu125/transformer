import os
import logging

class Logger(object):
    def __init__(self, name, level, log_path):
        fmt = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')

     
        self.logger = logging.getLogger(name)

        self.logger.setLevel(level)
        
        log_path = os.path.abspath(log_path)

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(fmt)
        file_handler.setLevel(level)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(fmt)
        stream_handler.setLevel(level)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

    def info(self, msg):
        self.logger.info(msg)

    def debug(self, meg):
        self.logger.debug(meg)

    def warning(self, msg):
        self.logger.warning(msg)
    
    def error(self, msg):
        self.logger.error(msg)
    