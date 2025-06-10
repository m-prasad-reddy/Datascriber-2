import logging
import logging.config
from pathlib import Path
import configparser
from typing import Optional
from config.utils import ConfigUtils, ConfigError

class LoggingSetup:
    """Manages logging configuration for the Datascriber system.

    Configures logging from logging_config.ini, supporting console, file, and notification logs.
    Provides component-specific loggers as singletons.

    Attributes:
        config_utils (ConfigUtils): Configuration utility instance.
        loggers (dict): Cache of logger instances.
        _instance (LoggingSetup): Singleton instance.
        _configured (bool): Flag to prevent redundant configuration.
    """

    _instance: Optional['LoggingSetup'] = None
    _configured: bool = False

    @classmethod
    def get_instance(cls, config_utils: Optional[ConfigUtils] = None) -> 'LoggingSetup':
        """Get the singleton instance of LoggingSetup.

        Args:
            config_utils (ConfigUtils, optional): Configuration utility instance.

        Returns:
            LoggingSetup: Singleton instance.

        Raises:
            ConfigError: If config_utils is missing on first initialization.
        """
        if cls._instance is None:
            if config_utils is None:
                raise ConfigError("ConfigUtils required for first initialization")
            cls._instance = cls(config_utils)
        return cls._instance

    def __init__(self, config_utils: ConfigUtils):
        """Initialize LoggingSetup.

        Args:
            config_utils (ConfigUtils): Configuration utility instance.

        Raises:
            ConfigError: If logging configuration fails.
        """
        if self._instance is not None:
            raise ConfigError("LoggingSetup is a singleton. Use get_instance().")
        self.config_utils = config_utils
        self.loggers = {}
        self._configure_logging()

    def _configure_logging(self) -> None:
        """Configure logging based on logging_config.ini.

        Raises:
            ConfigError: If configuration file is invalid or missing.
        """
        if self._configured:
            return
        try:
            config_path = self.config_utils.config_dir / "logging_config.ini"
            if not config_path.exists():
                self._create_default_logging_config(config_path)
            config = configparser.ConfigParser()
            config.read(config_path)
            logging.config.fileConfig(config, disable_existing_loggers=True)
            self._configured = True
        except Exception as e:
            raise ConfigError(f"Failed to configure logging: {str(e)}")

    def _create_default_logging_config(self, config_path: Path) -> None:
        """Create a default logging configuration file.

        Args:
            config_path (Path): Path to the logging configuration file.
        """
        config = configparser.ConfigParser()
        config['loggers'] = {'keys': 'root,datascriber,notifications'}
        config['handlers'] = {'keys': 'console,file,notification_file'}
        config['formatters'] = {'keys': 'standard'}
        config['logger_root'] = {
            'level': 'DEBUG',
            'handlers': 'console,file'
        }
        config['logger_datascriber'] = {
            'level': 'DEBUG',
            'handlers': 'console,file',
            'qualname': 'datascriber',
            'propagate': '0'
        }
        config['logger_notifications'] = {
            'level': 'INFO',
            'handlers': 'notification_file',
            'qualname': 'notifications',
            'propagate': '0'
        }
        config['handler_console'] = {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'args': '(sys.stdout,)'
        }
        config['handler_file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'standard',
            'args': f"('{self.config_utils.logs_dir / 'datascriber.log'}', 'a', 10485760, 5)"
        }
        config['handler_notification_file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'args': f"('{self.config_utils.logs_dir / 'notifications.log'}', 'a', 10485760, 5)"
        }
        config['formatter_standard'] = {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        }
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            config.write(f)

    def get_logger(self, name: str, component: str = "system") -> logging.Logger:
        """Get a logger instance for a specific component.

        Args:
            name (str): Logger name (e.g., 'cli', 'data_executor').
            component (str): Component identifier (e.g., 'system', datasource name).

        Returns:
            logging.Logger: Configured logger instance.
        """
        logger_name = f"datascriber.{name}.{component}"
        if logger_name not in self.loggers:
            self.loggers[logger_name] = logging.getLogger(logger_name)
        return self.loggers[logger_name]