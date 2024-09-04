import logging
from logging.handlers import RotatingFileHandler
import os


def setup_logger(name, log_file, level=logging.INFO):
    # Создаем директорию для логов, если она не существует
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')

    handler = RotatingFileHandler(log_file, maxBytes=10000000, backupCount=5)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


# Создаем логгеры для разных частей приложения
api_logger = setup_logger('api', os.path.join('logs', 'api.log'))
auth_logger = setup_logger('auth', os.path.join('logs', 'auth.log'))
classifier_logger = setup_logger('classifier', os.path.join('logs', 'classifier.log'))