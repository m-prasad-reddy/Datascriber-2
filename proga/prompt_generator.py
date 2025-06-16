import json
from typing import Dict, List, Optional
from pathlib import Path
from config.utils import ConfigUtils, ConfigError
from storage.storage_manager import StorageManager
from nlp.nlp_processor import NLPProcessor
from storage.db_manager import DBManager
from config.logging_setup import LoggingSetup
import traceback
from openai import AzureOpenAI

class PromptError(Exception):
    """Custom exception for prompt generation errors."""
    pass

class PromptGenerator:
    """Prompt generator for creating SQL query prompts in the Datascriber project.

    Generates system/user prompts for LLM-based Text-to-SQL, supports training data generation,
    and handles S3/SQL Server datasources dynamically.

    Attributes:
        config_utils (ConfigUtils): Configuration utility instance.
        logger (logging.Logger): System-wide logger.
        llm_config (Dict): LLM configuration from llm_config.json.
        azure_config (Dict): Azure configuration from azure_config.json.
        storage_manager (StorageManager): Storage manager instance.
        nlp_processor (NLPProcessor): NLP processor instance.
        default_mappings (Dict): Default synonym mappings from config.
        enable_component_logging (bool): Flag for component output logging.
        client (AzureOpenAI): Azure OpenAI client for SQL generation.
    """

    def __init__(self, config_utils: ConfigUtils):
        """Initialize PromptGenerator.

        Args:
            config_utils (ConfigUtils): Configuration utility instance.

        Raises:
            PromptError: If initialization fails.
        """
        self.config_utils = config_utils
        self.logger = LoggingSetup.get_logger(__name__)
        self.enable_component_logging = LoggingSetup.LOGGING_CONFIG.get("enable_component_logging", False)
        try:
            self.llm_config = self._load_llm_config()
            self.azure_config = self._load_azure_config()
            self.storage_manager = StorageManager(self.config_utils)
            self.nlp_processor = NLPProcessor(self.config_utils)
            self.default_mappings = self._load_default_mappings()
            self.client = AzureOpenAI(
                azure_endpoint=self.azure_config["azure_endpoint"],
                api_key=self.azure_config["api_key"],
                api_version=self.azure_config["api_version"],
                default_headers=self.azure_config.get("custom_auth_headers", {})
            )
            self.logger.debug("Initialized PromptGenerator with Azure OpenAI client")
            if self.enable_component_logging:
                print("Component Output: Initialized PromptGenerator")
        except (ConfigError, KeyError) as e:
            self.logger.error(f"Failed to initialize PromptGenerator: {str(e)}\n{traceback.format_exc()}")
            raise PromptError(f"Failed to initialize PromptGenerator: {str(e)}")

    def _load_llm_config(self) -> Dict:
        """Load LLM configuration from llm_config.json.

        Returns:
            Dict: LLM configuration.

        Raises:
            PromptError: If configuration loading fails.
        """
        try:
            config_path = self.config_utils.config_dir / "llm_config.json"
            with open(config_path, "r") as f:
                config = json.load(f)
            self.logger.debug(f"Loaded LLM configuration from {config_path}")
            if self.enable_component_logging:
                print(f"Component Output: Loaded LLM configuration from {config_path}")
            return config
        except (json.JSONDecodeError, FileNotFoundError) as e:
            self.logger.error(f"Failed to load llm_config.json: {str(e)}\n{traceback.format_exc()}")
            raise PromptError(f"Failed to load llm_config.json: {str(e)}")

    def _load_azure_config(self) -> Dict:
        """Load Azure configuration from azure_config.json.

        Returns:
            Dict: Azure configuration.

        Raises:
            PromptError: If configuration loading fails.
        """
        try:
            config_path = self.config_utils.config_dir / "azure_config.json"
            with open(config_path, "r") as f:
                config = json.load(f)
            self.logger.debug(f"Loaded Azure configuration from {config_path}")
            if self.enable_component_logging:
                print(f"Component Output: Loaded Azure configuration from {config_path}")
            return config
        except (json.JSONDecodeError, FileNotFoundError) as e:
            self.logger.error(f"Failed to load azure_config.json: {str(e)}\n{traceback.format_exc()}")
            raise PromptError(f"Failed to load azure_config.json: {str(e)}")

    def _load_default_mappings(self) -> Dict:
        """Load default synonym mappings from default_mappings.json.

        Returns:
            Dict: Default mappings.

        Raises:
            PromptError: If loading fails critically.
        """
        try:
            config_path = self.config_utils.config_dir / "default_mappings.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = json.load(f)
                self.logger.debug(f"Loaded default mappings from {config_path}")
                if self.enable_component_logging:
                    print(f"Component Output: Loaded default mappings from {config_path}")
                return config.get("common_mappings", {})
            self.logger.warning(f"default_mappings.json not found at {config_path}, using empty fallback")
            return {}
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse default_mappings.json: {str(e)}\n{traceback.format_exc()}")
            return {}

    def generate_system_prompt(self, datasource: Dict, schemas: List[str]) -> str:
        """Generate system prompt for LLM across multiple schemas.

        Args:
            datasource (Dict): Datasource configuration.
            schemas (List[str]): List of schema names.

        Returns:
            str: System prompt.

        Raises:
            PromptError: If generation fails.
        """
        if not schemas:
            self.logger.error("No schemas provided for system prompt generation")
            raise PromptError("No schemas provided")
        try:
            self.storage_manager._set_datasource(datasource)
            base_prompt = self.llm_config["prompt_settings"]["system_prompt"]
            file_type = datasource["type"]
            if file_type == "s3":
                file_type = self.storage_manager.file_type or "csv"
            prompt = (
                f"{base_prompt}\n"
                f"Datasource: {datasource['name']} ({file_type})\n"
            )
            for schema in schemas:
                try:
                    metadata = self._get_metadata(datasource, schema)
                    if not metadata.get("tables"):
                        self.logger.warning(f"No tables found for schema {schema}")
                        continue
                    filtered_metadata = {
                        "tables": [
                            table for table in metadata.get("tables", [])
                            if isinstance(table, dict) and "name" in table
                        ]
                    }
                    prompt += (
                        f"Schema: {schema}\n"
                        f"Metadata: {json.dumps(filtered_metadata, indent=2)}\n"
                    )
                except PromptError as e:
                    self.logger.warning(f"Skipping schema {schema} due to metadata error: {str(e)}")
                    try:
                        self.storage_manager.store_rejected_query(
                            datasource, "", schema, f"No metadata for schema {schema}", "system", "NO_METADATA"
                        )
                    except Exception as store_e:
                        self.logger.error(f"Failed to store rejected query for schema {schema}: {str(store_e)}\n{traceback.format_exc()}")
                    continue
            date_function = "YEAR(column)" if datasource["type"] == "sqlserver" else "EXTRACT(YEAR FROM column)"
            if datasource["type"] == "s3":
                prompt += (
                    f"Use DuckDB SQL for S3 queries. Access files using 'read_csv('s3://bucket/path/to/file.csv')' for CSV or "
                    f"'read_parquet('s3://bucket/path/to/file.parquet')' for Parquet. "
                    f"Table names should match the file names without extensions. "
                    f"Use {date_function} for dates, LOWER and LIKE for strings, SUM and AVG for numerics. "
                    f"Ensure SQL is valid for DuckDB."
                )
            else:
                prompt += (
                    f"Use {date_function} for dates, LOWER and LIKE for strings, "
                    f"SUM and AVG for numerics. Ensure SQL is valid for {file_type}."
                )
            self.logger.debug(f"Generated system prompt for schemas {schemas}, length: {len(prompt)}")
            if self.enable_component_logging:
                print(f"Component Output: Generated system prompt for schemas {schemas}, length {len(prompt)}")
            return prompt
        except (KeyError, json.JSONDecodeError) as e:
            self.logger.error(f"Failed to generate system prompt for schemas {schemas}: {str(e)}\n{traceback.format_exc()}")
            try:
                self.storage_manager.store_rejected_query(
                    datasource, "", schemas[0], f"System prompt generation failed: {str(e)}", "system", "PROMPT_ERROR"
                )
            except Exception as store_e:
                self.logger.error(f"Failed to store rejected query: {str(store_e)}\n{traceback.format_exc()}")
            raise PromptError(f"Failed to generate system prompt: {str(e)}")

    def _get_metadata(self, datasource: Dict, schema: str) -> Dict:
        """Fetch metadata for schema and datasource.

        Args:
            datasource (Dict): Datasource configuration.
            schema (str): Schema name.

        Returns:
            Dict: Metadata with tables and columns.

        Raises:
            PromptError: If metadata fetching fails.
        """
        try:
            metadata = self.config_utils.load_metadata(datasource["name"], [schema]).get(schema, {})
            if not metadata:
                self.logger.error(f"No metadata found for schema {schema} in datasource {datasource['name']}")
                raise PromptError(f"No metadata for schema {schema}")
            self.logger.debug(f"Fetched metadata for schema {schema} in datasource {datasource['name']}")
            if self.enable_component_logging:
                print(f"Component Output: Fetched metadata for schema {schema}")
            return metadata
        except ConfigError as e:
            self.logger.error(f"Failed to fetch metadata for schema {schema} in datasource {datasource['name']}: {str(e)}\n{traceback.format_exc()}")
            raise PromptError(f"Failed to fetch metadata: {str(e)}")

    def generate_user_prompt(self, datasource: Dict, nlq: str, schemas: List[str], entities: Optional[Dict] = None, prediction: Optional[Dict] = None, user_role: str = "user") -> str:
        """Generate user prompt for LLM across multiple schemas.

        Args:
            datasource (Dict): Datasource configuration.
            nlq (str): Natural language query.
            schemas (List[str]): List of schema names.
            entities (Optional[Dict]): Extracted entities (dates, names, objects, places).
            prediction (Optional[Dict]): Prediction result from TableIdentifier.
            user_role (str): User role (e.g., 'admin', 'user').

        Returns:
            str: User prompt.

        Raises:
            PromptError: If generation fails.
        """
        if not schemas:
            self.logger.error(f"No schemas provided for user prompt generation, NLQ: {nlq}")
            raise PromptError("No schemas provided")
        try:
            self.storage_manager._set_datasource(datasource)
            entities = entities or self.nlp_processor.process_query(nlq, schemas[0], datasource=datasource).get("entities", {})
            prediction = prediction or {"tables": [], "columns": [], "entities": entities, "ddl": ""}
            if not prediction.get("tables"):
                self.logger.warning(f"No tables in prediction for NLQ: {nlq}, using fallback")
                prediction["tables"] = self._get_related_tables(nlq, schemas[0], datasource, self._get_metadata(datasource, schemas[0]))
                prediction["columns"] = self._get_specific_columns(nlq, schemas[0], datasource, self._get_metadata(datasource, schemas[0]))
                prediction["ddl"] = ""
            max_length = self.llm_config["prompt_settings"]["max_prompt_length"]
            file_type = self.storage_manager.file_type or "csv"
            s3_path = datasource.get("s3_path", "")
            prompt = (
                f"User Query: {nlq}\n"
                f"Schemas: {', '.join(schemas)}\n"
                f"Datasource: {datasource['name']}\n"
                f"Entities: {json.dumps(entities, indent=2)}\n"
                f"Predicted Tables: {', '.join(prediction['tables'])}\n"
                f"Predicted Columns: {', '.join(prediction['columns'])}\n"
            )
            if prediction.get("ddl"):
                prompt += f"DDL:\n{prediction['ddl']}\n"
            else:
                metadata_added = False
                for schema in schemas:
                    try:
                        metadata = self._get_metadata(datasource, schema)
                        filtered_tables = [
                            table for table in metadata.get("tables", [])
                            if isinstance(table, dict) and table.get("name") in [t.split(".")[-1] for t in prediction["tables"]]
                        ]
                        if filtered_tables:
                            filtered_metadata = {"tables": filtered_tables}
                            schema_prompt = (
                                f"Schema: {schema}\n"
                                f"Metadata: {json.dumps(filtered_metadata, indent=2)}\n"
                            )
                            if len(prompt + schema_prompt) <= max_length:
                                prompt += schema_prompt
                                metadata_added = True
                            else:
                                self.logger.warning(f"Skipping metadata for schema {schema} due to length limit, NLQ: {nlq}")
                    except PromptError as e:
                        self.logger.warning(f"Skipping schema {schema} due to metadata error: {str(e)}, NLQ: {nlq}")
                        try:
                            self.storage_manager.store_rejected_query(
                                datasource, nlq, schema, f"No metadata for schema {schema}", "system", "NO_METADATA"
                            )
                        except Exception as store_e:
                            self.logger.error(f"Failed to store rejected query for schema {schema}, NLQ: {nlq}: {str(store_e)}\n{traceback.format_exc()}")
                        continue
                if not metadata_added:
                    self.logger.debug(f"No metadata added for NLQ: {nlq}, relying on prediction")
            date_function = "YEAR(column)" if datasource["type"] == "sqlserver" else "EXTRACT(YEAR FROM column)"
            if datasource["type"] == "s3":
                prompt += (
                    f"Generate a valid DuckDB SQL query for the S3 datasource. "
                    f"Access files using 'read_{'csv' if file_type == 'csv' else 'parquet'}('{s3_path}')'. "
                    f"Table names should match the file names without extensions. "
                    f"Use {date_function} for dates, LOWER and LIKE for strings, SUM and AVG for numerics. "
                    f"Return only the SQL query wrapped in ```sql\n```."
                )
            else:
                prompt += (
                    f"Generate a valid SQL query for the {datasource['type']} datasource. "
                    f"Use {date_function} for dates, LOWER and LIKE for strings, "
                    f"SUM and AVG for numerics."
                )
            if len(prompt) > max_length:
                self.logger.warning(f"Prompt exceeds max length {len(prompt)}, truncating to minimal, NLQ: {nlq}")
                prompt = (
                    f"User Query: {nlq}\n"
                    f"Schemas: {', '.join(schemas)}\n"
                    f"Datasource: {datasource['name']}\n"
                    f"Entities: {json.dumps(entities, indent=2)}\n"
                    f"Predicted Tables: {', '.join(prediction['tables'])}\n"
                    f"Generate a valid DuckDB SQL query."
                )
            self.logger.debug(f"Generated user prompt for NLQ: {nlq}, length: {len(prompt)}")
            if self.enable_component_logging:
                print(f"Component Output: Generated user prompt for NLQ '{nlq}', length {len(prompt)}")
            return prompt
        except (KeyError, json.JSONDecodeError) as e:
            self.logger.error(f"Failed to generate user prompt for NLQ {nlq}, schemas {schemas}: {str(e)}\n{traceback.format_exc()}")
            try:
                self.storage_manager.store_rejected_query(
                    datasource, nlq, schemas[0], f"User prompt generation failed: {str(e)}", "system", "PROMPT_ERROR"
                )
            except Exception as store_e:
                self.logger.error(f"Failed to store rejected query for NLQ {nlq}: {str(store_e)}\n{traceback.format_exc()}")
            raise PromptError(f"Failed to generate user prompt: {str(e)}")

    def _build_context(self, entities: Dict, schema: str, datasource: Dict, metadata: Dict, prediction: Dict) -> str:
        """Build context for user prompt.

        Args:
            entities (Dict): Extracted entities.
            schema (str): Schema name.
            datasource (Dict): Datasource configuration.
            metadata (Dict): Schema metadata.
            prediction (Dict): Prediction result.

        Returns:
            str: Context string.
        """
        try:
            context = []
            for key, values in entities.items():
                if values:
                    context.append(f"{key.capitalize()} detected: {', '.join(values)}")
            if prediction.get("tables"):
                context.append(f"Predicted tables: {', '.join(prediction['tables'])}")
            context.append(f"Use schema {schema} from datasource {datasource['name']}")
            if prediction.get("ddl"):
                context.append(f"DDL available for predicted tables")
            else:
                tables = [t.split(".")[-1] for t in prediction.get("tables", [])]
                if tables:
                    context.append(f"Available tables: {', '.join(tables)}")
            result = "; ".join(context)
            self.logger.debug(f"Built context for schema {schema}: {result}")
            if self.enable_component_logging:
                print(f"Component Output: Built context for schema {schema}, length {len(result)}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to build context for schema {schema}: {str(e)}\n{traceback.format_exc()}")
            try:
                self.storage_manager.store_rejected_query(
                    datasource, "", schema, f"Context build failed: {str(e)}", "system", "CONTEXT_ERROR"
                )
            except Exception as store_e:
                self.logger.error(f"Failed to store rejected query: {str(store_e)}\n{traceback.format_exc()}")
            return ""

    def generate_training_data(self, datasource: Dict, nlq: str, schema: str, entities: Dict, sql: str, prediction: Optional[Dict] = None, scenario_id: Optional[str] = None) -> Dict:
        """Generate a single training data row for storage in SQLite.

        Args:
            datasource (Dict): Datasource configuration.
            nlq (str): Natural language query.
            schema (str): Schema name.
            entities (Dict): Extracted entities.
            sql (str): Generated SQL query.
            prediction (Optional[Dict]): Prediction result from TableIdentifier.
            scenario_id (Optional[str]): Scenario ID for tracking.

        Returns:
            Dict: Training data row.

        Raises:
            PromptError: If generation fails.
        """
        try:
            metadata = self._get_metadata(datasource, schema)
            prediction = prediction or self.nlp_processor.process_query(nlq, schema, datasource=datasource)
            tables = prediction.get("tables", self._get_related_tables(nlq, schema, datasource, metadata))
            columns = prediction.get("columns", self._get_specific_columns(nlq, schema, datasource, metadata))
            placeholders = prediction.get("placeholders", self._generate_placeholders(entities.get("extracted_values", {})))
            row = {
                "db_source_type": datasource["type"],
                "db_name": datasource["name"],
                "user_query": nlq,
                "related_tables": json.dumps(tables),
                "specific_columns": json.dumps(columns),
                "relevant_sql": sql,
                "extracted_values": json.dumps(entities.get("extracted_values", {})),
                "placeholders": json.dumps(placeholders),
                "llm_sql": sql,
                "is_lsql_valid": True,
                "context_text1": self._build_context(entities, schema, datasource, metadata, prediction),
                "context_text2": "",
                "IS_SLM_TRAINED": False,
                "scenario_id": scenario_id or ""
            }
            self.logger.debug(f"Generated training data row for NLQ: {nlq}, schema: {schema}, scenario_id: {scenario_id}")
            if self.enable_component_logging:
                print(f"Component Output: Generated training data row for NLQ '{nlq}', scenario_id {scenario_id}")
            return row
        except (json.JSONDecodeError, KeyError) as e:
            self.logger.error(f"Failed to generate training data for NLQ {nlq}, schema {schema}: {str(e)}\n{traceback.format_exc()}")
            raise PromptError(f"Failed to generate training data: {str(e)}")

    def _get_related_tables(self, nlq: str, schema: str, datasource: Dict, metadata: Dict) -> List[str]:
        """Identify related tables for NLQ.

        Args:
            nlq (str): Natural language query.
            schema (str): Schema name.
            datasource (Dict): Datasource configuration.
            metadata (Dict): Schema metadata.

        Returns:
            List[str]: Related table names.
        """
        try:
            nlp_result = self.nlp_processor.process_query(nlq.lower(), schema, datasource=datasource)
            tokens = nlp_result.get("tokens", [])
            tables = []
            tables_data = metadata.get("tables", [])
            if not isinstance(tables_data, list):
                self.logger.warning(f"Invalid metadata: 'tables' is {type(tables_data)}, expected list")
                return []
            table_names = [table["name"] for table in tables_data if isinstance(table, dict) and "name" in table]
            for token in tokens:
                for key, syn_list in self.default_mappings.items():
                    if token.lower() == key or token.lower() in [s.lower() for s in syn_list]:
                        if key in table_names:
                            tables.append(f"{schema}.{key}")
            if not tables and table_names:
                tables.append(f"{schema}.{table_names[0]}")
            self.logger.debug(f"Identified related tables for NLQ '{nlq}': {tables}")
            if self.enable_component_logging:
                print(f"Component Output: Identified {len(tables)} related tables for NLQ '{nlq}'")
            return tables
        except Exception as e:
            self.logger.error(f"Failed to identify related tables for NLQ {nlq}, schema {schema}: {str(e)}\n{traceback.format_exc()}")
            return []

    def _get_specific_columns(self, nlq: str, schema: str, datasource: Dict, metadata: Dict) -> List[str]:
        """Identify specific columns for NLQ.

        Args:
            nlq (str): Natural language query.
            schema (str): Schema name.
            datasource (Dict): Datasource configuration.
            metadata (Dict): Schema metadata.

        Returns:
            List[str]: Specific column names.
        """
        try:
            nlp_result = self.nlp_processor.process_query(nlq.lower(), schema, datasource=datasource)
            tokens = nlp_result.get("tokens", [])
            columns = []
            tables_data = metadata.get("tables", [])
            if not isinstance(tables_data, list):
                self.logger.warning(f"Invalid metadata: 'tables' is {type(tables_data)}, expected list")
                return []
            for table in tables_data:
                if not isinstance(table, dict) or "columns" not in table:
                    continue
                for col in table.get("columns", []):
                    col_name = col["name"].lower()
                    if any(token in col_name for token in tokens):
                        columns.append(col["name"])
            if not columns and tables_data:
                first_table = next((t for t in tables_data if isinstance(t, dict) and "columns" in t), None)
                if first_table:
                    columns = [col["name"] for col in first_table.get("columns", [])[:2]]
            self.logger.debug(f"Identified specific columns for NLQ '{nlq}': {columns}")
            if self.enable_component_logging:
                print(f"Component Output: Identified {len(columns)} specific columns for NLQ '{nlq}'")
            return columns
        except Exception as e:
            self.logger.error(f"Failed to identify specific columns for NLQ {nlq}, schema {schema}: {str(e)}\n{traceback.format_exc()}")
            return []

    def _generate_placeholders(self, extracted_values: Dict) -> List[str]:
        """Generate placeholders for extracted values.

        Args:
            extracted_values (Dict): Extracted values from entities.

        Returns:
            List[str]: Placeholders for SQL query.
        """
        placeholders = ["?" for _ in extracted_values]
        self.logger.debug(f"Generated {len(placeholders)} placeholders")
        if self.enable_component_logging:
            print(f"Component Output: Generated {len(placeholders)} placeholders")
        return placeholders

    def save_training_data(self, datasource: Dict, training_rows: List[Dict]) -> None:
        """Save training data to SQLite via DBManager.

        Args:
            datasource (Dict): Datasource configuration.
            training_rows (List[Dict]): List of training data rows.

        Raises:
            PromptError: If saving fails.
        """
        try:
            db_manager = DBManager(self.config_utils)
            max_rows = self.llm_config["training_settings"].get("max_rows", 100)
            training_rows = training_rows[:max_rows]
            required_keys = [
                "db_source_type", "db_name", "user_query", "related_tables", "specific_columns",
                "relevant_sql", "extracted_values", "placeholders", "llm_sql", "is_lsql_valid",
                "context_text1", "context_text2", "IS_SLM_TRAINED", "scenario_id"
            ]
            for row in training_rows:
                missing_keys = [key for key in required_keys if key not in row or row[key] is None]
                if missing_keys:
                    self.logger.error(f"Missing keys in training row: {missing_keys}, row: {row}")
                    raise PromptError(f"Missing keys in training data: {missing_keys}")
            db_manager.store_training_data(datasource, training_rows)
            self.logger.info(f"Saved {len(training_rows)} training rows for datasource {datasource['name']}")
            if self.enable_component_logging:
                print(f"Component Output: Saved {len(training_rows)} training rows")
        except Exception as e:
            self.logger.error(f"Failed to save training data for datasource {datasource['name']}: {str(e)}\n{traceback.format_exc()}")
            raise PromptError(f"Failed to save training data: {str(e)}")

    def _extract_sql(self, response: str) -> str:
        """Extract SQL query from Azure OpenAI response.

        Args:
            response (str): Response text from Azure OpenAI.

        Returns:
            str: Extracted SQL query.
        """
        try:
            start_marker = "```sql"
            end_marker = "```"
            start_idx = response.find(start_marker)
            if start_idx != -1:
                start_idx += len(start_marker)
                end_idx = response.find(end_marker, start_idx)
                if end_idx != -1:
                    sql = response[start_idx:end_idx].strip()
                    if "SELECT" in sql.upper() or "WITH" in sql.upper():
                        self.logger.debug(f"Extracted SQL from response: {sql}")
                        return sql
                    else:
                        self.logger.warning(f"Extracted SQL lacks SELECT or WITH clause: {sql}")
                        return ""
            # Fallback: Try to find raw SQL without markers
            lines = response.split("\n")
            sql_lines = [line.strip() for line in lines if line.strip().upper().startswith(("SELECT", "WITH"))]
            if sql_lines:
                sql = "\n".join(sql_lines).strip()
                self.logger.debug(f"Fallback SQL extraction: {sql}")
                return sql
            self.logger.warning(f"No valid SQL found in response: {response}")
            return ""
        except Exception as e:
            self.logger.error(f"Failed to extract SQL from response: {str(e)}\n{traceback.format_exc()}")
            return ""

    def generate_sql(self, datasource: Dict, system_prompt: str, user_prompt: str, schemas: List[str], entities: Dict, tia_result: Dict, user_role: str = "user") -> str:
        """Generate SQL query using Azure OpenAI for the given prompt.

        Args:
            datasource (Dict): Datasource configuration.
            system_prompt (str): System prompt for LLM.
            user_prompt (str): User prompt for LLM.
            schemas (List[str]): List of schema names.
            entities (Dict): Extracted entities from NLPProcessor.
            tia_result (Dict): TableIdentifier prediction result.
            user_role (str): User role (e.g., 'admin', 'user').

        Returns:
            str: Generated SQL query.

        Raises:
            PromptError: If SQL generation fails.
        """
        if not schemas:
            self.logger.error("No schemas provided for SQL generation")
            raise PromptError("No schemas provided")
        try:
            response = self.client.chat.completions.create(
                model=self.llm_config["model_name"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1000  # Increased to prevent truncation
            )
            raw_response = response.choices[0].message.content
            self.logger.debug(f"Raw LLM response: {raw_response}")
            sql = self._extract_sql(raw_response)
            if not sql:
                self.logger.warning(f"No valid SQL generated for schemas {schemas}")
                sql = "# No valid SQL generated"
            self.logger.info(f"Generated SQL query for schemas {schemas}: {sql}")
            if self.enable_component_logging:
                print(f"Component Output: Generated SQL query for schemas {schemas}, length {len(sql)}")
            if user_role == "admin":
                print(f"Generated SQL: {sql}")
            return sql
        except Exception as e:
            self.logger.error(f"Failed to generate SQL for schemas {schemas}: {str(e)}\n{traceback.format_exc()}")
            print(f"Error: Failed to call Azure OpenAI: {str(e)}")
            try:
                self.storage_manager.store_rejected_query(
                    datasource, user_prompt, schemas[0], f"SQL generation failed: {str(e)}", "system", "LLM_ERROR"
                )
            except Exception as store_e:
                self.logger.error(f"Failed to store rejected query: {str(store_e)}\n{traceback.format_exc()}")
            raise PromptError(f"Failed to generate SQL: {str(e)}")