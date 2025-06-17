import logging
import logging.handlers
import os
from pathlib import Path

class LoggingSetup:
    # Configuration object (edit as needed)
    LOGGING_CONFIG = {
        "log_level": "DEBUG",  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
        # File and HTTP logging settings    
        "log_file": os.path.join("logs", "datascriber.log"),
        "enable_http_logging": False,
        "enable_component_logging": True
    }

    @staticmethod
    def setup_logging() -> None:
        """Initialize logging with configurable settings."""
        config = LoggingSetup.LOGGING_CONFIG

        # Set log level
        log_level = getattr(logging, config["log_level"].upper(), logging.INFO)

        # Create logger
        logger = logging.getLogger()
        logger.setLevel(log_level)

        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(
            "%(asctime)s\t%(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler
        try:
            # Compute absolute path relative to project root
            log_file = config["log_file"]
            log_file_path = Path(__file__).parent.parent / log_file
            # Create logs directory if it doesn't exist
            os.makedirs(log_file_path.parent, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                log_file_path, maxBytes=10*1024*1024, backupCount=5
            )
            file_handler.setLevel(log_level)
            file_formatter = logging.Formatter(
                "%(asctime)s\t%(levelname)s - %(name)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            logger.debug(f"File logging initialized to: {log_file_path}")
        except (OSError, ValueError) as e:
            logger.error(f"Failed to initialize file logging to {log_file}: {str(e)}")
            print(f"Error: File logging setup failed: {str(e)}")

        # Configure HTTP logging
        if not config["enable_http_logging"]:
            for http_logger in ["httpcore", "httpx", "openai._base_client"]:
                logging.getLogger(http_logger).setLevel(logging.WARNING)

        # Log configuration
        logger.info(f"Logging initialized: level={config['log_level']}, "
                    f"file={config['log_file']}, "
                    f"http_logging={config['enable_http_logging']}, "
                    f"component_logging={config['enable_component_logging']}")

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """Get a configured logger."""
        return logging.getLogger(name)