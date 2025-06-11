import logging
from typing import Optional
from config.utils import ConfigUtils

class LoggingSetup:
    """Configures system-wide logging for Datascriber."""
    _instance = None

    def __init__(self):
        """Initialize LoggingSetup."""
        self.loggers = {}

    @classmethod
    def get_instance(cls, config_utils: ConfigUtils, debug: bool = False) -> 'LoggingSetup':
        """Get singleton instance of LoggingSetup."""
        if cls._instance is None:
            instance = cls()
            log_level = logging.DEBUG if debug else logging.INFO
            logging.basicConfig(
                level=log_level,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler('logs/datascriber.log'),
                    logging.StreamHandler()
                ]
            )
            cls._instance = instance
        return cls._instance

    def get_logger(self, name: str, logger_type: str) -> logging.Logger:
        """Get a logger instance for a specific component."""
        logger_name = f"{logger_type}.{name}"
        if logger_name not in self.loggers:
            logger = logging.getLogger(logger_name)
            self.loggers[logger_name] = logger
        return self.loggers[logger_name]