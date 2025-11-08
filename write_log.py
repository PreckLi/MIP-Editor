import logging
import os
import datetime

def write_logger(args, message, log_dir):
    current_time = datetime.datetime.now().strftime('%Y%m%d')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file_path = os.path.join(log_dir, f'qwen2vl_mllmu.log')
    handler = logging.FileHandler(log_file_path, encoding='utf-8')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.info(f'{current_time}\nArgs: {args}\nMessage: {message}')
    handler.close()
    logger.removeHandler(handler)
