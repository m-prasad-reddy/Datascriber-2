import argparse
import sys
import json
import warnings
from pathlib import Path
from typing import Optional, Tuple
from config.utils import ConfigUtils, ConfigError
from config.logging_setup import LoggingSetup
from cli.interface import Interface
from storage.db_manager import DBManager, DBError
from storage.storage_manager import StorageManager, StorageError
from nlp.nlp_processor import NLPProcessor, NLPError
from core.orchestrator import Orchestrator
import platform
import importlib.metadata
import traceback

# Suppress importlib.metadata warnings if any
warnings.filterwarnings("ignore", category=DeprecationWarning)

__version__ = "1.1.0"

def validate_config(config_utils: ConfigUtils) -> None:
    """Validate required configuration files.

    Args:
        config_utils (ConfigUtils): Configuration utility instance.

    Raises:
        ConfigError: If validation fails.
    """
    logger = LoggingSetup.get_logger("main")
    try:
        config_files = [
            "db_configurations.json",
            "llm_config.json",
            "model_config.json",
            "azure_config.json",
            "aws_config.json",
            "synonym_config.json"
        ]
        for config_file in config_files:
            config_path = config_utils.config_dir / config_file
            if not config_path.exists():
                logger.error(f"Missing configuration file: {config_file}")
                raise ConfigError(f"Missing configuration file: {config_file}")
            with open(config_path, "r") as f:
                json.load(f)
            logger.debug(f"Validated configuration file: {config_file}")
        logger.debug("All configuration files validated successfully")
        if LoggingSetup.LOGGING_CONFIG.get("enable_component_logging"):
            print(f"Component Output: Validated configuration files: {config_files}")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file: {str(e)}\n{traceback.format_exc()}")
        raise ConfigError(f"Invalid JSON in configuration file: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to validate configurations: {str(e)}\n{traceback.format_exc()}")
        raise ConfigError(f"Failed to validate configurations: {str(e)}")

def initialize_components(config_utils: ConfigUtils, datasource_name: Optional[str] = None, schema_name: Optional[str] = None) -> Tuple[DBManager, StorageManager, NLPProcessor, Orchestrator]:
    """Initialize core components.

    Args:
        config_utils (ConfigUtils): Configuration utility instance.
        datasource_name (Optional[str]): Datasource to initialize.
        schema_name (Optional[str]): Schema to validate.

    Returns:
        Tuple: (db_manager, storage_manager, nlp_processor, orchestrator)

    Raises:
        ConfigError, DBError, StorageError, NLPError: If initialization fails.
    """
    logger = LoggingSetup.get_logger("main")
    try:
        logger.debug("Initializing core components")
        db_manager = DBManager(config_utils)
        storage_manager = StorageManager(config_utils)
        nlp_processor = NLPProcessor(config_utils)
        orchestrator = Orchestrator(config_utils, db_manager, storage_manager, nlp_processor)

        # Validate datasource if specified
        if datasource_name:
            datasources = config_utils.load_db_configurations().get("datasources", [])
            if not any(ds["name"] == datasource_name for ds in datasources):
                logger.error(f"Invalid datasource: {datasource_name}")
                raise ConfigError(f"Invalid datasource: {datasource_name}")
            datasource = next(ds for ds in datasources if ds["name"] == datasource_name)
            schemas = datasource["connection"].get("schemas", ["default"])
            if schema_name and schema_name not in schemas:
                logger.error(f"Invalid schema: {schema_name} for datasource {datasource_name}")
                raise ConfigError(f"Invalid schema: {schema_name}")
            for schema in schemas:
                if datasource["type"] == "sqlserver":
                    db_manager.validate_metadata(datasource, schema)
                elif datasource["type"] == "s3":
                    storage_manager.validate_metadata(datasource, schema)
            logger.info(f"Validated datasource {datasource_name} with schemas {schemas}")
            if LoggingSetup.LOGGING_CONFIG.get("enable_component_logging"):
                print(f"Component Output: Initialized datasource {datasource_name} with schemas {schemas}")

        logger.debug("Core components initialized successfully")
        if LoggingSetup.LOGGING_CONFIG.get("enable_component_logging"):
            print("Component Output: Core components initialized successfully")
        return db_manager, storage_manager, nlp_processor, orchestrator
    except (ConfigError, DBError, StorageError, NLPError) as e:
        logger.error(f"Failed to initialize components: {str(e)}\n{traceback.format_exc()}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during component initialization: {str(e)}\n{traceback.format_exc()}")
        raise ConfigError(f"Unexpected error during component initialization: {str(e)}")

def cleanup_components(db_manager: Optional[DBManager], storage_manager: Optional[StorageManager], nlp_processor: Optional[NLPProcessor]) -> None:
    """Cleanup resources.

    Args:
        db_manager (Optional[DBManager]): Database manager instance.
        storage_manager (Optional[StorageManager]): Storage manager instance.
        nlp_processor (Optional[NLPProcessor]): NLP processor instance.
    """
    logger = LoggingSetup.get_logger("main")
    try:
        if db_manager:
            db_manager.close_connections()
            logger.debug("Closed database connections")
        if storage_manager:
            storage_manager.clear_cache()
            logger.debug("Cleared storage cache")
        # Note: nlp_processor.clear_cache() removed due to missing method
        # Future: Implement clear_cache in NLPProcessor if needed
        logger.debug("Resource cleanup completed successfully")
        if LoggingSetup.LOGGING_CONFIG.get("enable_component_logging"):
            print("Component Output: Resource cleanup completed")
    except (DBError, StorageError) as e:
        logger.error(f"Failed to cleanup resources: {str(e)}\n{traceback.format_exc()}")

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

    db_manager = None
    storage_manager = None
    nlp_processor = None
    logger = None

    try:
        # Set debug logging if specified
        if args.debug:
            LoggingSetup.LOGGING_CONFIG["log_level"] = "DEBUG"

        # Initialize logging
        LoggingSetup.setup_logging()
        logger = LoggingSetup.get_logger("main")
        logger.info(f"Starting Datascriber v{__version__} on {platform.system()} {platform.release()}")

        # Log dependencies
        try:
            dependencies = [f"{dist.metadata['Name']}=={dist.version}" for dist in importlib.metadata.distributions()]
            logger.debug(f"Python: {sys.version}, Dependencies: {', '.join(dependencies)}")
            if LoggingSetup.LOGGING_CONFIG.get("enable_component_logging"):
                print(f"Component Output: Dependencies: {', '.join(dependencies)}")
        except Exception as e:
            logger.error(f"Failed to log dependencies: {str(e)}\n{traceback.format_exc()}")

        # Initialize ConfigUtils
        config_utils = ConfigUtils()

        # Validate configuration files
        validate_config(config_utils)

        # Initialize components
        logger.debug(f"Starting in {args.mode} mode")
        db_manager, storage_manager, nlp_processor, orchestrator = initialize_components(
            config_utils, args.datasource, args.schema
        )

        # Auto-select datasource for CLI mode
        if args.mode == "cli" and args.datasource:
            if orchestrator.select_datasource(args.datasource):
                config = config_utils.load_db_configurations()
                datasource = next((ds for ds in config["datasources"] if ds["name"] == args.datasource), None)
                if datasource:
                    schemas = datasource["connection"].get("schemas", ["default"])
                    if args.schema and args.schema not in schemas:
                        logger.error(f"Invalid schema: {args.schema}")
                        raise ConfigError(f"Invalid schema: {args.schema}")
                    if orchestrator.validate_metadata(datasource, schemas=schemas):
                        logger.info(f"Auto-selected datasource: {args.datasource}")
                        if LoggingSetup.LOGGING_CONFIG.get("enable_component_logging"):
                            print(f"Component Output: Auto-selected datasource {args.datasource}")
                    else:
                        logger.error(f"Metadata validation failed for datasource: {args.datasource}")
                        raise ConfigError("Invalid metadata")
                else:
                    logger.error(f"Datasource configuration not found: {args.datasource}")
                    raise ConfigError("Datasource configuration not found")

        # Run in specified mode
        if args.mode == "cli":
            logger.debug("Starting CLI mode")
            cli = Interface(config_utils, orchestrator)
            cli.run()
        elif args.mode == "batch":
            logger.info("Batch mode selected but not implemented")
            raise NotImplementedError("Batch mode is not yet implemented. Use --mode cli instead.")

    except (ConfigError, DBError, StorageError, NLPError) as e:
        if logger:
            logger.error(f"System error: {str(e)}\n{traceback.format_exc()}")
        print(f"Error: {str(e)}")
        sys.exit(1)
    except KeyboardInterrupt:
        if logger:
            logger.info("User interrupted execution")
        print("Exiting...")
        sys.exit(0)
    except Exception as e:
        if logger:
            logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)
    finally:
        if logger:
            cleanup_components(db_manager, storage_manager, nlp_processor)
            logger.info("Datascriber shutdown complete")
            if LoggingSetup.LOGGING_CONFIG.get("enable_component_logging"):
                print("Component Output: Datascriber shutdown complete")

if __name__ == "__main__":
    main()