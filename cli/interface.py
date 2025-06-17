import argparse
import json
from pathlib import Path
from typing import Optional, Dict, List
import shlex
from tabulate import tabulate
from datetime import datetime
import re
from config.utils import ConfigUtils, ConfigError
from core.orchestrator import Orchestrator
from config.logging_setup import LoggingSetup
from nlp.nlp_processor import NLPProcessor
import traceback
import contextlib
import sys
import os

class CLIError(Exception):
    """Custom exception for CLI-related errors."""
    pass

class Interface:
    """Command-line interface for the Datascriber system.

    Provides a menu-driven CLI with query mode for data users and admins.

    Attributes:
        config_utils (ConfigUtils): Configuration utility instance.
        orchestrator (Orchestrator): Orchestrator instance.
        logger: System logger.
        username (Optional[str]): Current user.
        datasource (Optional[Dict]): Selected datasource configuration.
        parser (argparse.ArgumentParser): Argument parser.
        llm_config (Dict): LLM configuration for validation.
        enable_component_logging (bool): Flag for component output logging.
    """

    def __init__(self, config_utils: ConfigUtils, orchestrator: Orchestrator):
        """Initialize Interface.

        Args:
            config_utils (ConfigUtils): Configuration utility instance.
            orchestrator (Orchestrator): Orchestrator instance.

        Raises:
            CLIError: If initialization fails.
        """
        self.config_utils = config_utils
        self.orchestrator = orchestrator
        self.logger = LoggingSetup.get_logger(__name__)
        self.enable_component_logging = LoggingSetup.LOGGING_CONFIG.get("enable_component_logging", False)
        try:
            self.username = None
            self.datasource = None
            self.parser = self._create_parser()
            self.llm_config = self._load_llm_config()
            self.logger.debug("Initialized CLI Interface")
            if self.enable_component_logging:
                print("Component Output: Initialized CLI Interface")
        except ConfigError as e:
            self.logger.error(f"Failed to initialize CLI: {str(e)}\n{traceback.format_exc()}")
            raise CLIError(f"Failed to initialize CLI: {str(e)}")

    def _load_llm_config(self) -> Dict:
        """Load LLM configuration for validation rules.

        Returns:
            Dict: LLM configuration.

        Raises:
            CLIError: If loading fails.
        """
        try:
            config = self.config_utils.load_llm_config()
            self.logger.debug("Loaded LLM config for validation")
            if self.enable_component_logging:
                print("Component Output: Loaded LLM config for validation")
            return config
        except ConfigError as e:
            self.logger.error(f"Failed to load llm_config.json: {str(e)}\n{traceback.format_exc()}")
            raise CLIError(f"Failed to load LLM config: {str(e)}")

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser.

        Returns:
            argparse.ArgumentParser: Configured parser.
        """
        parser = argparse.ArgumentParser(description="Datascriber CLI", add_help=False)
        subparsers = parser.add_subparsers(dest="command", help="Available commands")
        login_parser = subparsers.add_parser("login", help="Log in as a user")
        login_parser.add_argument("username", help="Username (admin/datauser)")
        ds_parser = subparsers.add_parser("select-datasource", help="Select a datasource")
        ds_parser.add_argument("name", help="Datasource name")
        query_parser = subparsers.add_parser("query", help="Submit an NLQ")
        query_parser.add_argument("nlq", nargs="+", type=str, help="Natural language query")
        subparsers.add_parser("refresh-metadata", help="Refresh metadata (admin only)")
        subparsers.add_parser("train-model", help="Train prediction model (admin only)")
        mode_parser = subparsers.add_parser("set-synonym-mode", help="Set synonym mode (admin only)")
        mode_parser.add_argument("mode", choices=["static", "dynamic"], help="Synonym mode")
        map_parser = subparsers.add_parser("map-failed-query", help="Map failed queries (admin only)")
        map_parser.add_argument("query_id", type=int, help="Query ID")
        map_parser.add_argument("sql", type=str, help="Corrected SQL query")
        map_parser.add_argument("--schema", default="default", help="Schema name")
        subparsers.add_parser("list-datasources", help="List available datasources")
        subparsers.add_parser("list-schemas", help="List available schemas")
        subparsers.add_parser("query-mode", help="Enter query mode (admin/datauser)")
        list_cols_parser = subparsers.add_parser("list-columns", help="List columns for a table")
        list_cols_parser.add_argument("table", help="Table name")
        list_cols_parser.add_argument("--schema", default="default", help="Schema name")
        notif_parser = subparsers.add_parser("manage-notifications", help="Manage notifications (view/resolve/delete/retry)")
        notif_parser.add_argument("action", choices=["view", "resolve", "delete", "retry"], help="Notification action")
        notif_parser.add_argument("id", nargs="?", type=int, help="Notification ID (required for resolve/delete/retry)")
        subparsers.add_parser("exit", help="Exit the CLI")
        return parser

    def run(self) -> None:
        """Run the main CLI loop.

        Raises:
            CLIError: If execution fails.
        """
        try:
            print("Welcome to Datascriber CLI. Type 'exit' to quit.")
            while True:
                try:
                    command = input("> ").strip()
                    if not command:
                        continue
                    args = self._parse_args_from_input(command)
                    if args.command == "exit":
                        self.logger.info("User exited CLI")
                        if self.enable_component_logging:
                            print("Component Output: Exiting CLI")
                        print("Exiting CLI...")
                        return
                    self.execute_command(args)
                except KeyboardInterrupt:
                    self.logger.info("User interrupted CLI")
                    if self.enable_component_logging:
                        print("Component Output: User interrupted CLI")
                    print("\nExiting...")
                    return
                except (argparse.ArgumentError, SystemExit, ValueError):
                    self.logger.warning(f"Invalid command: {command}\n{traceback.format_exc()}")
                    print(f"Invalid command: {command}. Use 'login', 'select-datasource', 'query', 'query-mode', etc.")
                except CLIError as e:
                    self.logger.error(f"Command error: {str(e)}\n{traceback.format_exc()}")
                    print(f"Error: {str(e)}")
        except Exception as e:
            self.logger.error(f"CLI execution failed: {str(e)}\n{traceback.format_exc()}")
            raise CLIError(f"CLI execution failed: {str(e)}")

    def _parse_args_from_input(self, command: str) -> argparse.Namespace:
        """Parse command input.

        Args:
            command (str): Raw command string.

        Returns:
            argparse.Namespace: Parsed arguments.

        Raises:
            argparse.ArgumentError: If parsing fails.
        """
        self.logger.debug(f"Parsing command: {command}")
        if self.enable_component_logging:
            print(f"Component Output: Parsing command: {command}")
        try:
            args = shlex.split(command)
            if not args:
                raise argparse.ArgumentError(None, "Empty command")
            return self.parser.parse_args(args)
        except ValueError:
            raise argparse.ArgumentError(None, "Invalid command syntax")

    def _validate_date_format(self, nlq: str) -> bool:
        """Validate date formats in the natural language query.

        Args:
            nlq (str): Natural language query.

        Returns:
            bool: True if valid or no dates, False if invalid.

        Raises:
            CLIError: If date format is invalid.
        """
        try:
            date_formats = self.llm_config["prompt_settings"]["validation"].get("date_format", [])
            if not date_formats:
                self.logger.debug("No date formats specified in llm_config")
                return True
            date_pattern = r'\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4}'
            matches = re.findall(date_pattern, nlq)
            if not matches:
                self.logger.debug("No date patterns found in NLQ")
                return True
            for matched_date in matches:
                for fmt in date_formats:
                    try:
                        python_fmt = fmt.replace("YYYY", "%Y").replace("MM", "%m").replace("DD", "%d")
                        datetime.strptime(matched_date, python_fmt)
                        self.logger.debug(f"Valid date format found in NLQ: {matched_date} ({fmt})")
                        return True
                    except ValueError:
                        continue
                error_msg = self.llm_config["prompt_settings"]["validation"].get("error_message", "Invalid date format")
                self.logger.error(f"Invalid date format in NLQ: {matched_date}")
                raise CLIError(error_msg)
            return True
        except KeyError as e:
            self.logger.warning(f"Invalid llm_config structure: {str(e)}. Skipping date validation.\n{traceback.format_exc()}")
            return True

    def execute_command(self, args: argparse.Namespace) -> None:
        """Execute a CLI command.

        Args:
            args (argparse.Namespace): Parsed arguments.

        Raises:
            CLIError: If execution fails.
        """
        self.logger.debug(f"Executing command: {args.command}")
        if self.enable_component_logging:
            print(f"Component Output: Executing command: {args.command}")
        try:
            if args.command == "login":
                self.login(args.username)
            elif args.command == "select-datasource":
                self.select_datasource(args.name)
            elif args.command == "query":
                nlq = " ".join(args.nlq)
                if self._validate_date_format(nlq):
                    self.submit_query(nlq)
            elif args.command == "refresh-metadata":
                self.refresh_metadata()
            elif args.command == "train-model":
                self.train_model()
            elif args.command == "set-synonym-mode":
                self.set_synonym_mode(args.mode)
            elif args.command == "map-failed-query":
                self.map_failed_query(args.query_id, args.sql, args.schema)
            elif args.command == "list-datasources":
                self.list_datasources()
            elif args.command == "list-schemas":
                self.list_schemas()
            elif args.command == "query-mode":
                self.enter_query_mode()
            elif args.command == "list-columns":
                self.list_columns(args.table, args.schema)
            elif args.command == "manage-notifications":
                self.manage_notifications(args.action, args.id)
            else:
                self.logger.warning(f"Unknown command: {args.command}")
                print("Unknown command")
        except CLIError as e:
            self.logger.error(f"Failed to execute command {args.command}: {str(e)}\n{traceback.format_exc()}")
            raise

    def login(self, username: str) -> None:
        """Log in a user.

        Args:
            username (str): Username.

        Raises:
            CLIError: If login fails.
        """
        if self.orchestrator.login(username):
            self.username = username
            self.orchestrator.reset_session()
            self.logger.info(f"User {username} logged in")
            if self.enable_component_logging:
                print(f"Component Output: Logged in as {username}")
            print(f"Logged in as {username}")
        else:
            self.logger.error(f"Login failed for username: {username}\n{traceback.format_exc()}")
            raise CLIError("Invalid username")

    def select_datasource(self, name: str) -> None:
        """Select a datasource configuration.

        Args:
            name (str): Datasource name.

        Raises:
            CLIError: If selection fails.
        """
        if not self.username:
            self.logger.error("No user logged in")
            raise CLIError("Please log in first")
        if self.orchestrator.select_datasource(name):
            config = self.config_utils.load_db_configurations()
            self.datasource = next((ds for ds in config["datasources"] if ds["name"] == name), None)
            if not self.datasource:
                self.logger.error(f"Datasource configuration not found: {name}\n{traceback.format_exc()}")
                raise CLIError("Datasource configuration not found")
            schemas = self.datasource["connection"].get("schemas", ["default"])
            self.logger.debug(f"Validating metadata for datasource {name} with schemas {schemas}")
            if not schemas and not self.datasource["connection"].get("tables"):
                self.logger.error(f"No schemas or tables configured for datasource: {name}")
                raise CLIError("No schemas or tables configured")
            if self.orchestrator.validate_metadata(self.datasource, schemas=schemas):
                self.logger.info(f"Selected datasource: {name}")
                if self.enable_component_logging:
                    print(f"Component Output: Selected datasource {name} with schemas {schemas}")
                print(f"Selected datasource: {name}")
                if self.username == "datauser":
                    self.enter_query_mode()
                else:
                    print("Type 'query-mode' to enter query mode or 'query <nlq>' to submit a query.")
            else:
                self.logger.error("Metadata validation failed")
                self.datasource = None
                if self.username == "datauser":
                    self.username = None
                    raise CLIError("Logged out due to invalid metadata")
                raise CLIError("Invalid or unavailable metadata")
        else:
            self.logger.error(f"Datasource selection failed: {name}\n{traceback.format_exc()}")
            raise CLIError("Invalid datasource")

    def enter_query_mode(self) -> None:
        """Enter query mode for continuous NLQ input.

        Raises:
            CLIError: If no user or datasource is selected.
        """
        if not self.username:
            self.logger.error("No user logged in")
            raise CLIError("Please log in first")
        if not self.datasource:
            self.logger.error("No datasource selected")
            raise CLIError("No datasource selected")
        self.logger.debug("Entered query mode")
        if self.enable_component_logging:
            print("Component Output: Entered query mode")
        print("Entered query mode. Type your query or 'exit' to return to main menu.")
        while True:
            try:
                query = input("Query> ").strip()
                if query.lower() == "exit":
                    self.logger.debug("Exited query mode")
                    if self.enable_component_logging:
                        print("Component Output: Exited query mode")
                    print("Exiting query mode.")
                    return
                if not query:
                    continue
                try:
                    with open(os.devnull, 'w') as devnull:
                        with contextlib.redirect_stderr(devnull):
                            args = self._parse_args_from_input(query)
                            if args.command in self.parser._subparsers._group_actions[0].choices:
                                self.logger.debug(f"Processing command in query mode: {query}")
                                if self.enable_component_logging:
                                    print(f"Component Output: Processing command in query mode: {query}")
                                self.execute_command(args)
                                continue
                except (argparse.ArgumentError, SystemExit, ValueError):
                    pass  # Treat as NLQ
                self.logger.debug(f"Processing NLQ in query mode: {query}")
                if self.enable_component_logging:
                    print(f"Component Output: Processing query: {query}")
                if self._validate_date_format(query):
                    self.submit_query(query)
            except KeyboardInterrupt:
                self.logger.info("User interrupted query mode")
                if self.enable_component_logging:
                    print("Component Output: User interrupted query mode")
                print("\nExiting query mode.")
                return
            except CLIError as e:
                self.logger.error(f"Query error: {str(e)}\n{traceback.format_exc()}")
                print(f"Error: {str(e)}")

    def submit_query(self, nlq: str) -> None:
        """Submit an NLQ across all configured schemas.

        Args:
            nlq (str): Natural language query.

        Raises:
            CLIError: If query fails.
        """
        if not self.username:
            self.logger.error("No user logged in")
            raise CLIError("No user logged in")
        if not self.datasource:
            self.logger.error("No datasource selected")
            raise CLIError("No datasource selected")
        schemas = self.datasource["connection"].get("schemas", ["default"])
        self.logger.debug(f"Processing NLQ '{nlq}' for schemas {schemas}")
        try:
            nlp_processor = NLPProcessor(self.config_utils)
            self.logger.debug(f"Initialized NLPProcessor for NLQ: {nlq}")
            entities = nlp_processor.process_query(nlq, schemas[0]).get("entities", {})
            self.logger.debug(f"Extracted entities for NLQ: {entities}")
            result = self.orchestrator.process_nlq(self.datasource, nlq, schemas, entities)
            if self.username == "admin" and result and result.get("predicted_tables"):
                predicted_tables = result["predicted_tables"]
                tia_result = result["tia_result"]
                entities = result["entities"]
                print(f"Predicted tables: {', '.join(predicted_tables)}")
                while True:
                    response = input("Are the predicted tables correct? (y/n): ").strip().lower()
                    if response in ["y", "n"]:
                        break
                    self.logger.warning(f"Invalid admin feedback response: {response}")
                    print("Please enter 'y' or 'n'.")
                is_correct = response == "y"
                validated_tables = predicted_tables if is_correct else []
                if not is_correct:
                    while True:
                        tables_input = input("Enter correct tables (comma-separated): ").strip()
                        if tables_input:
                            validated_tables = [t.strip() for t in tables_input.split(",")]
                            if all(t for t in validated_tables):
                                break
                            self.logger.warning(f"Invalid table input: {tables_input}")
                            print("Table names cannot be empty.")
                        else:
                            self.logger.warning("Empty table input")
                            print("Please enter at least one table.")
                self.logger.debug(f"Admin feedback: is_correct={is_correct}, validated_tables={validated_tables}")
                try:
                    self.orchestrator.process_admin_feedback(
                        self.datasource, nlq, schemas[0], entities, tia_result, validated_tables, is_correct
                    )
                    result = self.orchestrator.process_nlq(self.datasource, nlq, schemas, entities)
                except Exception as e:
                    self.logger.error(f"Failed to process admin feedback for NLQ '{nlq}': {str(e)}\n{traceback.format_exc()}")
                    self.orchestrator.notify_admin(self.datasource, nlq, schemas, f"Admin feedback processing failed: {str(e)}", entities)
                    raise CLIError(f"Failed to process admin feedback: {str(e)}")
            if result and result.get("sample_data"):
                self.logger.info(f"Query processed successfully: {nlq}")
                if self.enable_component_logging:
                    print(f"Component Output: Query '{nlq}' processed with tables {result['tables']}, "
                          f"columns {result['columns']}, SQL: {result['sql_query']}")
                print(f"Results for schemas {', '.join(schemas)}:")
                print(f"Tables: {', '.join(result['tables']) if result['tables'] else 'None'}")
                print(f"Columns: {', '.join(result['columns']) if result['columns'] else 'None'}")
                print(f"SQL Query: {result['sql_query']}")
                print("Sample Data:")
                if result["sample_data"]:
                    self.logger.debug(f"Formatting {len(result['sample_data'])} rows as table")
                    table = tabulate(
                        result["sample_data"],
                        headers="keys",
                        tablefmt="grid",
                        stralign="left",
                        numalign="right"
                    )
                    print(table)
                else:
                    print("No sample data available")
                print(f"Full results saved to: {result['csv_path']}")
                self.logger.info(f"Query results saved to: {result['csv_path']}")
                return
            message = "Query cannot be processed. Notified admin."
            self.logger.warning(f"No results for query: {nlq}")
            self.orchestrator.notify_admin(self.datasource, nlq, schemas, "No results returned", entities)
            print(message)
        except Exception as e:
            self.logger.error(f"Failed to process NLQ '{nlq}': {str(e)}\n{traceback.format_exc()}")
            self.orchestrator.notify_admin(self.datasource, nlq, schemas, str(e), entities)
            raise CLIError(f"Failed to process query: {str(e)}")

    def refresh_metadata(self) -> None:
        """Refresh metadata for all schemas (admin only).

        Raises:
            CLIError: If execution fails.
        """
        if not self.username or self.username != "admin":
            self.logger.error("Admin privileges required")
            raise CLIError("Admin privileges required")
        if not self.datasource:
            self.logger.error("No datasource selected")
            raise CLIError("No datasource selected")
        try:
            schemas = self.datasource["connection"].get("schemas", ["default"])
            self.logger.debug(f"Refreshing metadata for schemas {schemas}")
            if not schemas and not self.datasource["connection"].get("tables"):
                self.logger.error("No schemas or tables configured in db_configurations.json")
                raise CLIError("No schemas or tables configured")
            if self.orchestrator.refresh_metadata(self.datasource, schemas):
                self.logger.info(f"Metadata refreshed for schemas: {schemas}")
                if self.enable_component_logging:
                    print(f"Component Output: Metadata refreshed for schemas {schemas}")
                print(f"Metadata refreshed for schemas: {', '.join(schemas)}")
            else:
                self.logger.error(f"Metadata refresh failed for schemas: {schemas}\n{traceback.format_exc()}")
                raise CLIError("Metadata refresh failed")
        except KeyError as e:
            self.logger.error(f"Invalid datasource configuration: {str(e)}\n{traceback.format_exc()}")
            raise CLIError(f"Invalid datasource configuration: {str(e)}")

    def train_model(self) -> None:
        """Train the model (admin only).

        Raises:
            CLIError: If execution fails.
        """
        if not self.username or self.username != "admin":
            self.logger.error("Admin privileges required")
            raise CLIError("Admin privileges required")
        if not self.datasource:
            self.logger.error("No datasource selected")
            raise CLIError("No datasource selected")
        try:
            if self.orchestrator.train_model(self.datasource):
                self.logger.info("Trained prediction model")
                if self.enable_component_logging:
                    print("Component Output: Prediction model trained")
                print("Model training completed")
            else:
                self.logger.error(f"Model training failed\n{traceback.format_exc()}")
                raise CLIError("Model training failed")
        except ConfigError as e:
            self.logger.error(f"Failed to train model: {str(e)}\n{traceback.format_exc()}")
            raise CLIError(f"Failed to train model: {str(e)}")

    def set_synonym_mode(self, mode: str) -> None:
        """Set synonym mode (admin only).

        Args:
            mode (str): Synonym mode ('static' or 'dynamic').

        Raises:
            CLIError: If execution fails.
        """
        if not self.username or self.username != "admin":
            self.logger.error("Admin privileges required")
            raise CLIError("Admin privileges required")
        try:
            if self.orchestrator.set_synonym_mode(mode):
                self.logger.info(f"Set synonym mode to: {mode}")
                if self.enable_component_logging:
                    print(f"Component Output: Synonym mode set to {mode}")
                print(f"Synonym mode set to: {mode}")
            else:
                self.logger.error(f"Failed to set synonym mode\n{traceback.format_exc()}")
                raise CLIError("Failed to set synonym mode")
        except ConfigError as e:
            self.logger.error(f"Failed to set synonym mode: {str(e)}\n{traceback.format_exc()}")
            raise CLIError(f"Failed to set synonym mode: {str(e)}")

    def map_failed_query(self, query_id: int, sql: str, schema: str) -> None:
        """Map failed queries (admin only).

        Args:
            query_id (int): Query ID.
            sql (str): Corrected SQL query.
            schema (str): Schema name.

        Raises:
            CLIError: If execution fails.
        """
        if not self.username or self.username != "admin":
            self.logger.error("Admin privileges required")
            raise CLIError("Admin privileges required")
        if not self.datasource:
            self.logger.error("No datasource selected")
            raise CLIError("No datasource selected")
        try:
            rejected_queries = self.orchestrator.get_rejected_queries(self.datasource)
            selected_query = next((q for q in rejected_queries if q["id"] == query_id), None)
            if not selected_query:
                self.logger.error(f"Invalid query ID: {query_id}")
                raise CLIError("Invalid query ID")
            if self.orchestrator.map_failed_query(self.datasource, selected_query["query"], sql, schema):
                self.orchestrator.update_rejected_query(self.datasource, query_id, "mapped")
                self.logger.info(f"Mapped query ID {query_id}")
                if self.enable_component_logging:
                    print(f"Component Output: Query ID {query_id} mapped with SQL: {sql}")
                print(f"Query ID {query_id} mapped successfully")
            else:
                self.logger.error(f"Failed to map query ID: {query_id}\n{traceback.format_exc()}")
                raise CLIError("Failed to map query")
        except Exception as e:
            self.logger.error(f"Failed to map failed query: {str(e)}\n{traceback.format_exc()}")
            raise CLIError(f"Failed to map failed query: {str(e)}")

    def list_datasources(self) -> None:
        """List available datasources.

        Raises:
            CLIError: If execution fails.
        """
        try:
            config = self.config_utils.load_db_configurations()
            datasources = [ds["name"] for ds in config["datasources"]]
            if not datasources:
                self.logger.info("No datasources found")
                print("No datasources available.")
                return
            self.logger.info(f"Listed {len(datasources)} datasources")
            if self.enable_component_logging:
                print(f"Component Output: Listed datasources: {datasources}")
            print("Available datasources:")
            for ds in datasources:
                print(f"- {ds}")
        except ConfigError as e:
            self.logger.error(f"Failed to list datasources: {str(e)}\n{traceback.format_exc()}")
            raise CLIError(f"Failed to list datasources: {str(e)}")

    def list_schemas(self) -> None:
        """List available schemas for the selected datasource.

        Raises:
            CLIError: If execution fails.
        """
        if not self.datasource:
            self.logger.error("No datasource selected")
            raise CLIError("No datasource selected")
        try:
            schemas = self.datasource["connection"].get("schemas", ["default"])
            if not schemas:
                self.logger.info("No schemas found")
                print("No schemas configured.")
                return
            self.logger.info(f"Listed {len(schemas)} schemas")
            if self.enable_component_logging:
                print(f"Component Output: Listed schemas: {schemas}")
            print("Available schemas:")
            for schema in schemas:
                print(f"- {schema}")
        except KeyError as e:
            self.logger.error(f"Invalid datasource configuration: {str(e)}\n{traceback.format_exc()}")
            raise CLIError(f"Invalid datasource configuration: {str(e)}")

    def list_columns(self, table: str, schema: str) -> None:
        """List columns for a specified table.

        Args:
            table (str): Table name.
            schema (str): Schema name.

        Raises:
            CLIError: If execution fails.
        """
        if not self.datasource:
            self.logger.error("No datasource selected")
            raise CLIError("No datasource selected")
        try:
            metadata = self.orchestrator.get_metadata(self.datasource, schema)
            tables = metadata.get("tables", {})
            if table not in tables:
                self.logger.error(f"Table {table} not found in schema {schema}")
                raise CLIError(f"Table {table} not found")
            columns = tables[table].get("columns", [])
            if not columns:
                self.logger.info(f"No columns found for table {table}")
                print(f"No columns found for table {table}.")
                return
            self.logger.info(f"Listed {len(columns)} columns for table {table}")
            if self.enable_component_logging:
                print(f"Component Output: Listed columns for table {table}: {[col['name'] for col in columns]}")
            print(f"Columns for table {table} in schema {schema}:")
            for col in columns:
                print(f"- {col['name']} ({col['type']})")
        except Exception as e:
            self.logger.error(f"Failed to list columns: {str(e)}\n{traceback.format_exc()}")
            raise CLIError(f"Failed to list columns: {str(e)}")

    def manage_notifications(self, action: str, notification_id: Optional[int]) -> None:
        """Manage notifications (view/resolve/delete/retry).

        Args:
            action (str): Action to perform (view/resolve/delete/retry).
            notification_id (Optional[int]): Notification ID for resolve/delete/retry.

        Raises:
            CLIError: If execution fails.
        """
        if not self.username or self.username != "admin":
            self.logger.error("Admin privileges required")
            raise CLIError("Admin privileges required")
        if not self.datasource:
            self.logger.error("No datasource selected")
            raise CLIError("No datasource selected")
        try:
            if action == "view":
                rejected_queries = self.orchestrator.get_rejected_queries(self.datasource)
                if not rejected_queries:
                    self.logger.info("No notifications found")
                    print("No notifications available.")
                    return
                self.logger.info(f"Viewed {len(rejected_queries)} notifications")
                if self.enable_component_logging:
                    print(f"Component Output: Viewed notifications: {[q['id'] for q in rejected_queries]}")
                print("Notifications:")
                for query in rejected_queries:
                    print(f"ID: {query['id']}, Query: {query['query']}, Reason: {query['reason']}, Time: {query['timestamp']}")
            elif action in ["resolve", "delete", "retry"]:
                if not notification_id:
                    self.logger.error("Notification ID required")
                    raise CLIError("Notification ID required")
                if action == "resolve":
                    self.orchestrator.update_rejected_query(self.datasource, notification_id, "resolved")
                    self.logger.info(f"Resolved notification ID {notification_id}")
                    if self.enable_component_logging:
                        print(f"Component Output: Resolved notification ID {notification_id}")
                    print(f"Notification ID {notification_id} resolved.")
                elif action == "delete":
                    self.orchestrator.update_rejected_query(self.datasource, notification_id, "deleted")
                    self.logger.info(f"Deleted notification ID {notification_id}")
                    if self.enable_component_logging:
                        print(f"Component Output: Deleted notification ID {notification_id}")
                    print(f"Notification ID {notification_id} deleted.")
                elif action == "retry":
                    rejected_queries = self.orchestrator.get_rejected_queries(self.datasource)
                    query = next((q for q in rejected_queries if q["id"] == notification_id), None)
                    if not query:
                        self.logger.error(f"Invalid notification ID: {notification_id}")
                        raise CLIError("Invalid notification ID")
                    self.submit_query(query["query"])
                    self.orchestrator.update_rejected_query(self.datasource, notification_id, "retried")
                    self.logger.info(f"Retried notification ID {notification_id}")
                    if self.enable_component_logging:
                        print(f"Component Output: Retried notification ID {notification_id}")
                    print(f"Notification ID {notification_id} retried.")
        except Exception as e:
            self.logger.error(f"Failed to manage notifications: {str(e)}\n{traceback.format_exc()}")
            raise CLIError(f"Failed to manage notifications: {str(e)}")