import argparse
import json
from pathlib import Path
from typing import Optional, Dict, List
import shlex
from tabulate import tabulate
from datetime import datetime
import re
import logging
from config.utils import ConfigUtils, ConfigError
from core.orchestrator import Orchestrator
from nlp.nlp_processor import NLPProcessor
from storage.db_manager import DBManager, DBError

class CLIError(Exception):
    """Custom exception for CLI-related errors."""
    pass

class Interface:
    """Command-line interface for the Datascriber system.

    Provides a menu-driven CLI with query mode for data users and admins.

    Attributes:
        config_utils (ConfigUtils): Configuration utility instance.
        orchestrator (Orchestrator): Orchestrator instance.
        logger (logging.Logger): CLI logger.
        username (Optional[str]): Current user.
        datasource (Optional[Dict]): Selected datasource configuration.
        parser (argparse.ArgumentParser): Argument parser.
        llm_config (Dict): LLM configuration for validation.
    """

    def __init__(self, config_utils: ConfigUtils, orchestrator: Orchestrator, logger: logging.Logger):
        """Initialize Interface.

        Args:
            config_utils (ConfigUtils): Configuration utility instance.
            orchestrator (Orchestrator): Orchestrator instance.
            logger (logging.Logger): System logger.

        Raises:
            CLIError: If initialization fails.
        """
        self.config_utils = config_utils
        self.orchestrator = orchestrator
        self.logger = logger
        try:
            self.username = None
            self.datasource = None
            self.parser = self._create_parser()
            self.llm_config = self._load_llm_config()
            self.logger.debug("Initialized CLI Interface")
        except ConfigError as e:
            self.logger.error(f"Failed to initialize CLI: {str(e)}")
            raise CLIError(f"Failed to initialize CLI: {str(e)}")

    def _load_llm_config(self) -> Dict:
        """Load LLM configuration for validation rules.

        Returns:
            Dict: LLM configuration.

        Raises:
            CLIError: If loading fails.
        """
        try:
            config_path = self.config_utils.config_dir / "llm_config.json"
            with open(config_path, "r") as f:
                config = json.load(f)
            self.logger.debug("Loaded LLM config for validation")
            return config
        except (json.JSONDecodeError, FileNotFoundError) as e:
            self.logger.error(f"Failed to load llm_config.json: {str(e)}")
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
        map_parser.add_argument("--schema", default="default", help="Schema name")
        subparsers.add_parser("list-datasources", help="List available datasources")
        subparsers.add_parser("list-schemas", help="List available schemas")
        subparsers.add_parser("query-mode", help="Enter query mode (admin/datauser)")
        list_cols_parser = subparsers.add_parser("list-columns", help="List columns for a table")
        list_cols_parser.add_argument("table", help="Table name")
        list_cols_parser.add_argument("--schema", default="schema", help="Schema name")
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
                        print("Exiting CLI...")
                        self.logger.info("User exited CLI")
                        return
                    self.execute_command(args)
                except KeyboardInterrupt:
                    print("\nExiting...")
                    self.logger.info("User interrupted CLI")
                    return
                except (argparse.ArgumentError, ValueError) as e:
                    print(f"Invalid command: {str(e)}. Use 'login', 'select-datasource', 'query', etc.")
                    self.logger.warning(f"Invalid command: {command}")
                except CLIError as e:
                    print(f"Error: {str(e)}")
                    self.logger.error(f"Command error: {str(e)}")
        except Exception as e:
            self.logger.error(f"CLI execution failed: {str(e)}")
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
            bool: True if valid, False if invalid.

        Raises:
            CLIError: If date format is invalid.
        """
        try:
            date_formats = self.llm_config["prompt_settings"]["validation"].get("date_formats", [])
            if not date_formats:
                return True  # No validation required
            for fmt in date_formats:
                pattern = fmt.get("pattern")
                strftime = fmt.get("strftime")
                matches = re.findall(pattern, nlq)
                for matched_date in matches:
                    try:
                        datetime.strptime(matched_date, strftime)
                        self.logger.debug(f"Valid date format found in NLQ: {matched_date}")
                        return True
                    except ValueError:
                        continue
                if matches:
                    error_msg = self.llm_config["prompt_settings"]["validation"].get("error_message", "Invalid date format")
                    self.logger.error(f"Invalid date format in NLQ: {matches}")
                    raise CLIError(error_msg)
            return True  # No dates found
        except KeyError as e:
            self.logger.error(f"Invalid llm_config structure: {str(e)}")
            raise CLIError(f"Invalid LLM configuration: {str(e)}")

    def execute_command(self, args: argparse.Namespace) -> None:
        """Execute a CLI command.

        Args:
            args (argparse.Namespace): Parsed arguments.

        Raises:
            CLIError: If execution fails.
        """
        self.logger.debug(f"Executing command: {args.command}")
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
                self.map_failed_query(args.schema)
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
                print("Unknown command")
                self.logger.warning(f"Unknown command: {args.command}")
        except CLIError as e:
            self.logger.error(f"Failed to execute command {args.command}: {str(e)}")
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
            print(f"Logged in as {username}")
            self.logger.debug(f"User {username} logged in")
        else:
            self.logger.error(f"Login failed for username: {username}")
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
                self.logger.error(f"Datasource configuration not found: {name}")
                raise CLIError("Datasource configuration not found")
            schemas = self.datasource["connection"].get("schemas", ["default"])
            self.logger.debug(f"Validating metadata for datasource {name} with schemas {schemas} (type: {type(schemas)})")
            if not schemas and not self.datasource["connection"].get("tables"):
                self.logger.error(f"No schemas or tables configured for datasource: {name}")
                raise CLIError("No schemas or tables configured")
            if self.orchestrator.validate_metadata(self.datasource, schemas=schemas):
                print(f"Selected datasource: {name}")
                self.logger.debug(f"Selected datasource: {name}")
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
            self.logger.error(f"Datasource selection failed: {name}")
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
        print("Entered query mode. Type your query or 'exit' to return to main menu.")
        while True:
            try:
                query = input("Query> ").strip()
                if query.lower() == "exit":
                    print("Exiting query mode.")
                    self.logger.debug("Exited query mode")
                    return
                if not query:
                    continue
                if self._validate_date_format(query):
                    self.submit_query(query)
            except KeyboardInterrupt:
                print("\nExiting query mode.")
                self.logger.debug("User interrupted query mode")
                return
            except CLIError as e:
                print(f"Error: {str(e)}")
                self.logger.error(f"Query error: {str(e)}")

    def submit_query(self, nlq: str) -> None:
        """Submit an NLQ across all configured schemas.

        Args:
            nlq (str): Natural language query.

        Raises:
            CLIError: If query fails.
        """
        nlp_processor = NLPProcessor(self.config_utils, self.logger)
        if not self.username:
            self.logger.error("No user logged in")
            raise CLIError("No user logged in")
        if not self.datasource:
            self.logger.error("No datasource selected")
            raise CLIError("No datasource selected")
        schemas = self.datasource["connection"].get("schemas", ["default"])
        self.logger.debug(f"Processing NLQ '{nlq}' for schemas {schemas} (type: {type(schemas)})")
        try:
            entities = nlp_processor.process_query(nlq, schemas[0]).get("entities", {})
            result = self.orchestrator.process_nlq(self.datasource, nlq, schemas=schemas, entities=entities)
            if self.username == "admin" and result and result.get("predicted_tables"):
                # Handle admin feedback for table predictions
                predicted_tables = result["predicted_tables"]
                tia_result = result["tia_result"]
                entities = result["entities"]
                print(f"Predicted tables: {', '.join(predicted_tables)}")
                while True:
                    response = input("Are the predicted tables correct? (y/n): ").strip().lower()
                    if response in ["y", "n"]:
                        break
                    print("Please enter 'y' or 'n'.")
                    self.logger.warning(f"Invalid admin feedback response: {response}")
                is_correct = response == "y"
                validated_tables = predicted_tables if is_correct else []
                if not is_correct:
                    while True:
                        tables_input = input("Enter correct tables (comma-separated): ").strip()
                        if tables_input:
                            validated_tables = [t.strip() for t in tables_input.split(",")]
                            if all(t for t in validated_tables):
                                break
                            print("Table names cannot be empty.")
                            self.logger.warning(f"Invalid table input: {tables_input}")
                        else:
                            print("Please enter at least one table.")
                            self.logger.warning("Empty table input")
                self.logger.debug(f"Admin feedback: is_correct={is_correct}, validated_tables={validated_tables}")
                # Debug Orchestrator instance before calling process_admin_feedback
                self.logger.debug(f"Orchestrator type: {type(self.orchestrator)}, methods: {dir(self.orchestrator)}")
                # Process admin feedback and resume query processing
                try:
                    self.orchestrator.process_admin_feedback(
                        self.datasource, nlq, schemas[0], entities, tia_result, validated_tables, is_correct
                    )
                    # Re-run NLQ with validated tables
                    tia_result["tables"] = validated_tables
                    result = self.orchestrator.process_nlq(self.datasource, nlq, schemas=schemas, entities=entities)
                except Exception as e:
                    self.logger.error(f"Failed to process admin feedback for NLQ '{nlq}': {str(e)}")
                    self.orchestrator.notify_admin(self.datasource, nlq, schemas, f"Admin feedback processing failed: {str(e)}", entities)
                    raise CLIError(f"Failed to process admin feedback: {str(e)}")
            if result and result.get("sample_data"):
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
        except (ImportError, ConfigError) as e:
            self.logger.error(f"Failed to process NLQ: {str(e)}")
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
            self.logger.debug(f"Refreshing metadata for schemas {schemas} (type: {type(schemas)})")
            if not schemas and not self.datasource["connection"].get("tables"):
                raise CLIError("No schemas or tables configured in db_configurations.json")
            if self.orchestrator.refresh_metadata(self.datasource, schemas):
                print(f"Metadata refreshed for schemas: {', '.join(schemas)}")
                self.logger.debug(f"Refreshed metadata for schemas: {schemas}")
            else:
                self.logger.error(f"Metadata refresh failed for schemas: {schemas}")
                raise CLIError(f"Metadata refresh failed")
        except KeyError as e:
            self.logger.error(f"Invalid datasource configuration: {str(e)}")
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
                print("Model training completed")
                self.logger.info("Trained prediction model")
            else:
                self.logger.error("Model training failed")
                raise CLIError("Model training failed")
        except ConfigError as e:
            self.logger.error(f"Failed to train model: {str(e)}")
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
                print(f"Synonym mode set to: {mode}")
                self.logger.info(f"Set synonym mode to: {mode}")
            else:
                self.logger.error("Failed to set synonym mode")
                raise CLIError("Failed to set synonym mode")
        except ConfigError as e:
            self.logger.error(f"Failed to set synonym mode: {str(e)}")
            raise CLIError(f"Failed to set synonym mode: {str(e)}")

    def map_failed_query(self, schema: str) -> None:
        """Map failed queries (admin only).

        Args:
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
            db_manager = DBManager(self.config_utils, self.logger)
            rejected_queries = db_manager.get_rejected_queries(self.datasource)
            if not rejected_queries:
                print("No rejected queries found.")
                self.logger.info("No rejected queries to map")
                return
            for query in rejected_queries:
                print(f"Query ID: {query['id']}, NLQ: {query['query']}, Reason: {query['reason']}")
            query_id = int(input("Enter query ID to map (or 0 to cancel): "))
            if query_id == 0:
                self.logger.debug("Cancelled query mapping")
                return
            selected_query = next((q for q in rejected_queries if q["id"] == query_id), None)
            if not selected_query:
                self.logger.error(f"Invalid query ID: {query_id}")
                raise CLIError("Invalid query ID")
            corrected_sql = input("Enter corrected SQL query: ").strip()
            if self.orchestrator.map_failed_query(self.datasource, selected_query["query"], corrected_sql, schema):
                print(f"Query ID {query_id} mapped successfully")
                db_manager.update_rejected_query(self.datasource, query_id, "mapped")
                self.logger.info(f"Mapped query ID {query_id}")
            else:
                self.logger.error(f"Failed to map query ID: {query_id}")
                raise CLIError("Failed to map query")
        except (DBError, ValueError) as e:
            self.logger.error(f"Failed to map failed query: {str(e)}")
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
                print("No datasources available.")
                self.logger.info("No datasources found")
                return
            print("Available datasources:")
            for ds in datasources:
                print(f"- {ds}")
            self.logger.info(f"Listed {len(datasources)} datasources")
        except ConfigError as e:
            self.logger.error(f"Failed to list datasources: {str(e)}")
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
                print("No schemas configured.")
                self.logger.info("No schemas found")
                return
            print("Available schemas:")
            for schema in schemas:
                print(f"- {schema}")
            self.logger.info(f"Listed {len(schemas)} schemas")
        except KeyError as e:
            self.logger.error(f"Invalid datasource configuration: {str(e)}")
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
            db_manager = DBManager(self.config_utils, self.logger)
            metadata = db_manager.get_metadata(self.datasource, schema)
            tables = metadata.get("tables", {})
            if table not in tables:
                self.logger.error(f"Table {table} not found in schema {schema}")
                raise CLIError(f"Table {table} not found")
            columns = tables[table].get("columns", [])
            if not columns:
                print(f"No columns found for table {table}.")
                self.logger.info(f"No columns found for table {table}")
                return
            print(f"Columns for table {table} in schema {schema}:")
            for col in columns:
                print(f"- {col['name']} ({col['type']})")
            self.logger.info(f"Listed {len(columns)} columns for table {table}")
        except DBError as e:
            self.logger.error(f"Failed to list columns: {str(e)}")
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
            db_manager = DBManager(self.config_utils, self.logger)
            if action == "view":
                rejected_queries = db_manager.get_rejected_queries(self.datasource)
                if not rejected_queries:
                    print("No notifications available.")
                    self.logger.info("No notifications found")
                    return
                print("Notifications:")
                for query in rejected_queries:
                    print(f"ID: {query['id']}, Query: {query['query']}, Reason: {query['reason']}, Time: {query['timestamp']}")
                self.logger.info(f"Viewed {len(rejected_queries)} notifications")
            elif action in ["resolve", "delete", "retry"]:
                if not notification_id:
                    self.logger.error("Notification ID required")
                    raise CLIError("Notification ID required")
                if action == "resolve":
                    db_manager.update_rejected_query(self.datasource, notification_id, "resolved")
                    print(f"Notification ID {notification_id} resolved.")
                    self.logger.info(f"Resolved notification ID {notification_id}")
                elif action == "delete":
                    db_manager.delete_rejected_query(self.datasource, notification_id)
                    print(f"Notification ID {notification_id} deleted.")
                    self.logger.info(f"Deleted notification ID {notification_id}")
                elif action == "retry":
                    rejected_queries = db_manager.get_rejected_queries(self.datasource)
                    query = next((q for q in rejected_queries if q["id"] == notification_id), None)
                    if not query:
                        self.logger.error(f"Invalid notification ID: {notification_id}")
                        raise CLIError("Invalid notification ID")
                    self.submit_query(query["query"])
                    db_manager.update_rejected_query(self.datasource, notification_id, "retried")
                    print(f"Notification ID {notification_id} retried.")
                    self.logger.info(f"Retried notification ID {notification_id}")
        except DBError as e:
            self.logger.error(f"Failed to manage notifications: {str(e)}")
            raise CLIError(f"Failed to manage notifications: {str(e)}")