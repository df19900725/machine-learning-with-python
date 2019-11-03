import logging


def get_logger():
    # logger_format = '%(asctime)-%(message)'
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger()
    logger.setLevel('DEBUG')
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    return logger

