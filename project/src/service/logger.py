import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name: str = "bank_conversion_service") -> logging.Logger:
    """
    Настраивает логгер для сервиса.
    
    Args:
        name: Имя логгера
    
    Returns:
        Настроенный логгер
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Формат логов
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s %(name)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Консольный вывод
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


# Глобальный логгер
logger = setup_logger()