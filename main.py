import argparse
import sys
import json
from pathlib import Path
from typing import Optional
from config.utils import ConfigUtils, ConfigError
from config.logging_setup import LoggingSetup
from cli.interface import Interface
from storage.db_manager import DBManager, DBError
from storage.storage_manager import StorageManager, StorageError
from nlp.nlp_processor import NLPProcessor, NLPError
from core.orchestrator import Orchestrator
import platform
import pkg_resources

__version__ = "1.1.0"

def validate_config(config_utils: ConfigUtils, logger) -> None:
    """Validate required configuration files.

    Args:
        config_utils (ConfigUtils): Configuration utility instance.
        logger: System logger.

    Raises:
        ConfigError: If validation fails.
    """
    config_files = [
        "db_configurations.json",
        "llm_config.json",
        "model_config.json",
        "azure_config.json",
        "aws_config.json",
        "synonym_config.json"
    ]
    for config_file in config_files:
        try:
            config_path = config_utils.config_dir / config_file
            if not config_path.exists():
                logger.error(f"Missing configuration file: {config_file}")
                raise ConfigError(f"Missing configuration file: {config_file}")
            with open(config_path, "r") as f:
                json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {config_file}: {str(e)}")
            raise ConfigError(f"Invalid JSON in {config_file}: {str(e)}")

def initialize_components(config_utils: ConfigUtils, logger, datasource_name: Optional[str] = None):
    """Initialize core components.

    Args:
        config_utils (ConfigUtils): Configuration utility instance.
        logger: System logger.
        datasource_name (Optional[str]): Datasource to initialize.

    Returns:
        Tuple: (db_manager, storage_manager, nlp_processor, orchestrator)

    Raises:
        ConfigError, DBError, StorageError, NLPError: If initialization fails.
    """
    try:
        db_manager = DBManager(config_utils)
        storage_manager = StorageManager(config_utils)
        nlp_processor = NLPProcessor(config_utils)
        orchestrator = Orchestrator(config_utils, db_manager, storage_manager, nlp_processor)

        # Validate datasource if specified
        if datasource_name:
            datasources = config_utils.load_db_config().get("datasources", [])
            if not any(ds["name"] == datasource_name for ds in datasources):
                logger.error(f"Invalid datasource: {datasource_name}")
                raise ConfigError(f"Invalid datasource: {datasource_name}")
            datasource = next(ds for ds in datasources if ds["name"] == datasource_name)
            if datasource["type"] == "sqlserver":
                db_manager.validate_metadata(datasource, datasource["connection"]["schemas"][0])
            elif datasource["type"] == "s3":
                storage_manager.validate_metadata(datasource, datasource["connection"]["schemas"][0])

        return db_manager, storage_manager, nlp_processor, orchestrator
    except (ConfigError, DBError, StorageError, NLPError) as e:
        logger.error(f"Failed to initialize components: {str(e)}")
        raise

def cleanup_components(db_manager: DBManager, storage_manager: StorageManager, nlp_processor: NLPProcessor, logger):
    """Cleanup resources.

    Args:
        db_manager (DBManager): Database manager instance.
        storage_manager (StorageManager): Storage manager instance.
        nlp_processor (NLPProcessor): NLP processor instance.
        logger: System logger.
    """
    try:
        db_manager.close_connections()
        storage_manager.clear_cache()
        nlp_processor.clear_cache()
        logger.debug("Cleaned up resources")
    except (DBError, StorageError, NLPError) as e:
        logger.error(f"Failed to cleanup resources: {str(e)}")

def main():
    """Main entry point for the Datascriber system."""
    parser = argparse.ArgumentParser(description="Datascriber Text-to-SQL System")
    parser.add_argument("--datasource", type=str, help="Specify datasource name")
    parser.add_argument("--schema", type=str, help="Specify schema name")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--version", action="store_true", help="Show version and exit")
    parser.add_argument("--mode", choices=["cli", "batch"], default="cli", help="Run mode (cli or batch)")
    args = parser.parse_args()

    if args.version:
        print(f"Datascriber v{__version__}")
        sys.exit(0)

    try:
        # Initialize ConfigUtils and LoggingSetup
        config_utils = ConfigUtils()
        logging_setup = LoggingSetup.get_instance(config_utils, debug=args.debug)
        logger = logging_setup.get_logger("main", "system")

        # Log system information
        logger.info(f"Starting Datascriber v{__version__} on {platform.system()} {platform.release()}")
        logger.debug(f"Python: {sys.version}, Dependencies: {', '.join(f'{d.key}=={d.version}' for d in pkg_resources.working_set)}")

        # Validate configuration files
        validate_config(config_utils, logger)

        # Initialize components
        db_manager, storage_manager, nlp_processor, orchestrator = initialize_components(
            config_utils, logger, args.datasource
        )

        # Run in specified mode
        if args.mode == "cli":
            logger.debug("Starting CLI mode")
            cli = Interface(config_utils, orchestrator)
            cli.run(datasource=args.datasource, schema=args.schema)
        elif args.mode == "batch":
            logger.debug("Starting batch mode")
            # Placeholder for batch processing
            logger.info("Batch mode not implemented yet")
            raise NotImplementedError("Batch mode is not implemented")

    except (ConfigError, DBError, StorageError, NLPError) as e:
        logger.error(f"System error: {str(e)}")
        print(f"Error: {str(e)}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("User interrupted execution")
        print("Exiting...")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)
    finally:
        if 'db_manager' in locals():
            cleanup_components(db_manager, storage_manager, nlp_processor, logger)
        logger.info("Datascriber shutdown complete")

if __name__ == "__main__":
    main()