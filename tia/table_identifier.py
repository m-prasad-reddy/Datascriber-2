import json
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
from collections import OrderedDict
from config.utils import ConfigUtils, ConfigError
from openai import AzureOpenAI
from nlp.nlp_processor import NLPProcessor
from storage.db_manager import DBManager
import httpx
import traceback
import re
from config.logging_setup import LoggingSetup

class TIAError(Exception):
    """Custom exception for Table Identifier Agent errors."""
    pass

class TableIdentifier:
    """Table Identifier Agent for mapping NLQs to database schema elements.

    Uses semantic matching with embeddings for table/column prediction. Supports
    static/dynamic synonyms, manual/bulk training, and model generation.

    Attributes:
        config_utils (ConfigUtils): Configuration utility instance.
        logger (logging.Logger): System-wide logger.
        synonym_mode (str): 'static' or 'dynamic' synonym handling mode.
        model_type (str): Embedding model type (e.g., 'azure-openai').
        model_name (str): Model name (e.g., 'text-embedding-3-small').
        embedding_deployment_name (str): Deployment name for embeddings.
        confidence_threshold (float): Prediction confidence threshold.
        model_path (Path): Path to model file.
        loaded_model (Optional[Dict]): Loaded model data.
        default_mappings (Dict): Default synonym mappings from config.
        generating_model (bool): Flag to prevent recursive model generation.
        _embedding_cache (OrderedDict): LRU cache for storing embeddings.
        enable_component_logging (bool): Flag for component output logging.
    """

    def __init__(self, config_utils: ConfigUtils):
        """Initialize TableIdentifier.

        Args:
            config_utils (ConfigUtils): Configuration utility instance.

        Raises:
            TIAError: If initialization fails.
        """
        self.config_utils = config_utils
        self.logger = LoggingSetup.get_logger(__name__)
        self.enable_component_logging = LoggingSetup.LOGGING_CONFIG.get("enable_component_logging", False)
        self.generating_model = False
        try:
            self.datasource = None
            self.synonym_mode = self._load_synonym_mode()
            model_config = self._load_model_config()
            azure_config = self.config_utils.load_azure_config()
            self.model_type = model_config.get("model_type", "azure-openai")
            self.model_name = model_config.get("model_name", "text-embedding-3-small")
            self.embedding_deployment_name = azure_config.get(
                "embedding_deployment_name",
                model_config.get("embedding_deployment_name", "embedding-model")
            )
            self.confidence_threshold = model_config.get("confidence_threshold", 0.7)
            self.model_path = None
            self.loaded_model = None
            self.default_mappings = self._load_default_mappings()
            self._embedding_cache = OrderedDict()  # LRU cache
            self._embedding_cache_max_size = 1000  # Max cache size
            self.logger.debug(
                f"Initialized TableIdentifier with model_type={self.model_type}, "
                f"model_name={self.model_name}, embedding_deployment_name={self.embedding_deployment_name}, "
                f"synonym_mode={self.synonym_mode}"
            )
            if self.enable_component_logging:
                print("Component Output: Initialized TableIdentifier")
        except Exception as e:
            self.logger.error(f"Failed to initialize TableIdentifier: {str(e)}\n{traceback.format_exc()}")
            raise TIAError(f"Failed to initialize TableIdentifier: {str(e)}")

    def _load_default_mappings(self) -> Dict:
        """Load default synonym mappings from default_mappings.json.

        Returns:
            Dict: Default mappings.

        Raises:
            TIAError: If loading fails critically.
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

    def _set_datasource(self, datasource: Dict) -> None:
        """Set and validate datasource configuration without triggering model generation.

        Args:
            datasource (Dict): Datasource configuration.

        Raises:
            TIAError: If validation fails.
        """
        required_keys = ["name", "type", "connection"]
        if not all(key in datasource for key in required_keys):
            self.logger.error("Missing required keys in datasource configuration")
            raise TIAError("Missing required keys")
        self.datasource = datasource
        self.model_path = self.config_utils.models_dir / f"model_{datasource['name']}.json"
        self._load_model()
        self.logger.debug(f"Set datasource: {datasource['name']}")
        if self.enable_component_logging:
            print(f"Component Output: Set datasource {datasource['name']}")

    def _load_synonym_mode(self) -> str:
        """Load synonym mode from configuration.

        Returns:
            str: 'static' or 'dynamic'.

        Raises:
            TIAError: If configuration loading fails.
        """
        try:
            config_path = self.config_utils.config_dir / "synonym_config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = json.load(f)
                mode = config.get("synonym_mode", "static")
                if mode not in ["static", "language", "dynamic"]:
                    self.logger.warning(f"Invalid synonym mode {mode}, defaulting to static")
                    return "static"
                self.logger.debug(f"Loaded synonym mode: {mode}")
                if self.enable_component_logging:
                    print(f"Component Output: Loaded synonym mode {mode}")
                return mode
            return "static"
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse synonym config: {str(e)}\n{traceback.format_exc()}")
            raise TIAError(f"Failed to load synonym config: {str(e)}")

    def _load_model_config(self) -> Dict:
        """Load model configuration.

        Returns:
            Dict: Model configuration.

        Raises:
            TIAError: If configuration loading fails.
        """
        try:
            config_path = self.config_utils.config_dir / "model_config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = json.load(f)
                self.logger.debug(f"Loaded model config from {config_path}")
                if self.enable_component_logging:
                    print(f"Component Output: Loaded model config from {config_path}")
                return config
            return {
                "model_type": "azure-openai",
                "model_name": "text-embedding-3-small",
                "embedding_deployment_name": "embedding-model",
                "confidence_threshold": 0.7
            }
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse model config: {str(e)}\n{traceback.format_exc()}")
            raise TIAError(f"Failed to load model config: {str(e)}")

    def _load_model(self) -> None:
        """Load model if available.

        Raises:
            TIAError: If model loading fails.
        """
        if not self.model_path or not self.model_path.exists():
            self.logger.info(f"No model found at {self.model_path}")
            self.loaded_model = None
            return
        try:
            with open(self.model_path, "r") as f:
                self.loaded_model = json.load(f)
            required = ["queries", "tables", "columns", "embeddings"]
            for key in required:
                if key not in self.loaded_model:
                    self.logger.error(f"Invalid model: missing '{key}'")
                    self.loaded_model = None
                    return
            self.loaded_model["embeddings"] = np.array(self.loaded_model["embeddings"])
            if self.loaded_model["embeddings"].size == 0:
                self.logger.warning(f"Model at {self.model_path} has empty embeddings")
                self.loaded_model = None
                return
            self.logger.debug(f"Loaded model from {self.model_path}")
            if self.enable_component_logging:
                print(f"Component Output: Loaded model from {self.model_path}")
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Failed to load model {self.model_path}: {str(e)}\n{traceback.format_exc()}")
            raise TIAError(f"Failed to load model: {str(e)}")

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        try:
            self._embedding_cache.clear()
            self.logger.debug("Cleared embedding cache")
            if self.enable_component_logging:
                print("Component Output: Cleared embedding cache")
        except Exception as e:
            self.logger.error(f"Failed to clear embedding cache: {str(e)}\n{traceback.format_exc()}")

    def _get_metadata(self, schemas: List[str]) -> Dict:
        """Fetch metadata for schemas.

        Args:
            schemas (List[str]): Schema names.

        Returns:
            Dict: Metadata dictionary.

        Raises:
            TIAError: If metadata fetching fails.
        """
        try:
            if not self.datasource:
                self.logger.error("Datasource not set")
                raise TIAError("Datasource not set")
            metadata = self.config_utils.load_metadata(self.datasource["name"], schemas)
            for schema in schemas:
                if schema not in metadata:
                    self.logger.warning(f"No metadata found for schema {schema}")
                    metadata[schema] = {"tables": []}
            self.logger.debug(f"Fetched metadata for schemas {schemas}")
            if self.enable_component_logging:
                print(f"Component Output: Fetched metadata for schemas {schemas}")
            return metadata
        except ConfigError as e:
            self.logger.error(f"Failed to fetch metadata for schemas {schemas}: {str(e)}\n{traceback.format_exc()}")
            raise TIAError(f"Failed to fetch metadata: {str(e)}")

    def _load_synonyms(self, schema: str) -> Dict[str, List[str]]:
        """Load synonyms from metadata and default mappings.

        Args:
            schema (str): Schema name.

        Returns:
            Dict[str, List[str]]: Synonym mappings.

        Raises:
            TIAError: If synonym loading fails.
        """
        try:
            metadata = self._get_metadata([schema])
            synonyms = {}
            for table in metadata.get(schema, {}).get("tables", []):
                if not isinstance(table, dict) or "name" not in table:
                    self.logger.warning(f"Invalid table entry in metadata: {table}")
                    continue
                table_name = table.get("name")
                synonyms[table_name] = table.get("synonyms", []) + self.default_mappings.get(table_name.lower(), [])
                for column in table.get("columns", []):
                    if not isinstance(column, dict) or "name" not in column:
                        self.logger.warning(f"Invalid column entry in table {table_name}: {column}")
                        continue
                    col_key = f"{table_name}.{column['name']}"
                    synonyms[col_key] = column.get("synonyms", [])
            self.logger.debug(f"Loaded synonyms for schema {schema}: {len(synonyms)} entries")
            if self.enable_component_logging:
                print(f"Component Output: Loaded {len(synonyms)} synonyms for schema {schema}")
            return synonyms
        except TIAError as e:
            self.logger.error(f"Failed to load synonyms for schema {schema}: {str(e)}\n{traceback.format_exc()}")
            raise

    def identify_tables(self, datasource: Dict, nlq: str, schemas: List[str]) -> Dict:
        """Identify tables and columns for an NLQ.

        Args:
            datasource (Dict): Datasource configuration.
            nlq (str): Natural language query.
            schemas (List[str]): Schema names.

        Returns:
            Dict: Identification result.

        Raises:
            TIAError: If identification fails critically.
        """
        self.logger.debug(f"Identifying tables for NLQ '{nlq}' with schemas {schemas}")
        try:
            if not schemas:
                self.logger.error(f"No schemas provided for NLQ: {nlq}")
                raise TIAError("No schemas provided")
            self._set_datasource(datasource)
            # Generate model if it doesn't exist
            if not self.model_path.exists():
                self.logger.info(f"No model found at {self.model_path}, generating default")
                self.generate_model(datasource)
            nlp_processor = NLPProcessor(self.config_utils)
            result = None
            if self.loaded_model:
                result = self._predict_with_model(nlq, schemas[0], nlp_processor)
            if not result:
                result = self._predict_with_mappings(nlq, schemas, nlp_processor)
            if not result or not result.get("tables"):
                self.logger.warning(f"No tables identified for NLQ: {nlq}, using fallback")
                result = self._fallback_prediction(nlq, schemas, nlp_processor)
            if result and result.get("tables"):
                self.logger.info(f"Identification successful for NLQ: {nlq}, tables: {result['tables']}")
                if self.enable_component_logging:
                    print(f"Component Output: Identified tables {result['tables']} for NLQ '{nlq}'")
                return result
            self.logger.error(f"Failed to identify tables for NLQ: {nlq}")
            try:
                db_manager = DBManager(self.config_utils)
                db_manager.store_rejected_query(
                    datasource, nlq, schemas[0], "Unable to process request", "system", "TIA_ERROR"
                )
            except Exception as store_e:
                self.logger.error(f"Failed to store rejected query for NLQ {nlq}: {str(store_e)}\n{traceback.format_exc()}")
            return {
                "tables": [],
                "columns": [],
                "extracted_values": {},
                "placeholders": [],
                "entities": {},
                "ddl": "",
                "conditions": {},
                "sql": None
            }
        except Exception as e:
            self.logger.error(f"Identification failed for NLQ '{nlq}': {str(e)}\n{traceback.format_exc()}")
            raise TIAError(f"Identification failed: {str(e)}")

    def _predict_with_model(self, nlq: str, schema: str, nlp_processor: NLPProcessor) -> Optional[Dict]:
        """Identify using model embeddings.

        Args:
            nlq (str): Natural language query.
            schema (str): Schema name.
            nlp_processor (NLPProcessor): NLP processor instance.

        Returns:
            Optional[Dict]: Identification result or None.
        """
        try:
            nlq_embedding = self._encode_query(nlq)
            if self.loaded_model["embeddings"].size == 0:
                self.logger.warning(f"Empty embeddings in model for datasource {self.datasource['name']}, falling back to metadata-based prediction")
                return None
            self.logger.debug(f"NLQ embedding shape: {nlq_embedding.shape}, model embeddings shape: {self.loaded_model['embeddings'].shape}")
            similarities = np.dot(nlq_embedding, self.loaded_model["embeddings"].T) / (
                np.linalg.norm(nlq_embedding) * np.linalg.norm(self.loaded_model["embeddings"], axis=1)
            )
            max_sim_idx = np.argmax(similarities)
            max_sim_score = float(similarities[max_sim_idx])
            if max_sim_score >= self.confidence_threshold:
                query = self.loaded_model["queries"][max_sim_idx]
                tables = self.loaded_model["tables"][max_sim_idx]
                columns = self.loaded_model["columns"][max_sim_idx]
                nlp_result = nlp_processor.process_query(nlq, schema, datasource=self.datasource)
                # Validate tables against metadata
                metadata = self._get_metadata([schema])
                valid_tables = [
                    t for t in tables
                    if any(td.get("name") == t.split(".")[-1] for td in metadata.get(schema, {}).get("tables", []))
                ]
                if not valid_tables:
                    self.logger.warning(f"No valid tables found in model prediction for NLQ: {nlq}")
                    return None
                result = {
                    "tables": valid_tables,
                    "columns": columns,
                    "extracted_values": nlp_result.get("extracted_values", {}),
                    "placeholders": ["?" for _ in nlp_result.get("extracted_values", {})],
                    "entities": nlp_result.get("entities", {}),
                    "ddl": self._generate_ddl([t.split(".")[-1] for t in valid_tables], schema),
                    "conditions": self._extract_conditions(nlp_result),
                    "sql": self._get_stored_sql(query)
                }
                if self.enable_component_logging:
                    print(f"Component Output: Model predicted tables {valid_tables} for NLQ '{nlq}' with score {max_sim_score}")
                return result
            self.logger.debug(f"Model confidence too low: {max_sim_score} for NLQ: {nlq}")
            return None
        except (ValueError, TypeError) as e:
            self.logger.error(f"Model identification error for NLQ '{nlq}': {str(e)}\n{traceback.format_exc()}")
            return None

    def _predict_with_mappings(self, nlq: str, schemas: List[str], nlp_processor: NLPProcessor) -> Optional[Dict]:
        """Identify using metadata and NLP.

        Args:
            nlq (str): Natural language query.
            schemas (List[str]): Schema names.
            nlp_processor (NLPProcessor): NLP processor instance.

        Returns:
            Optional[Dict]: Identification result or None.
        """
        try:
            nlp_result = nlp_processor.process_query(nlq, schemas[0], datasource=self.datasource)
            tokens = nlp_result.get("tokens", [])
            extracted_values = nlp_result.get("extracted_values", {})
            entities = nlp_result.get("entities", {})
            primary_entity = entities.get("objects", [None])[0]  # e.g., 'customers'
            metadata = self._get_metadata(schemas)
            synonyms = {schema: self._load_synonyms(schema) for schema in schemas}
            result = {
                "tables": [],
                "columns": [],
                "extracted_values": extracted_values,
                "placeholders": ["?" for _ in extracted_values],
                "entities": entities,
                "ddl": "",
                "conditions": self._extract_conditions(nlp_result),
                "sql": None
            }
            # Calculate table scores
            table_scores = {}
            token_mappings = []
            valid_tables = {(schema, t["name"]) for schema in schemas for t in metadata.get(schema, {}).get("tables", [])}
            for token in tokens:
                mapped_term = nlp_processor.map_synonyms(token, synonyms[schemas[0]], schemas[0], self.datasource)
                token_mappings.append(f"Token: {token} -> Mapped: {mapped_term}")
                for schema in schemas:
                    schema_data = metadata.get(schema, {})
                    schema_boost = 0.3 if (primary_entity and token.lower() == primary_entity.lower() and token.lower() in schema.lower()) else 0.0
                    for table in schema_data.get("tables", []):
                        if not isinstance(table, dict) or "name" not in table:
                            self.logger.warning(f"Invalid table entry in metadata: {table}")
                            continue
                        table_name = table.get("name")
                        full_table = f"{schema}.{table_name}"
                        if (schema, table_name) not in valid_tables:
                            continue
                        table_synonyms = synonyms[schema].get(table_name, [])
                        score = 0.0
                        if mapped_term.lower() == table_name.lower():
                            score += 1.5  # Prioritize exact matches
                        elif (mapped_term.lower() in table_name.lower() or
                              any(mapped_term.lower() in s.lower() for s in table_synonyms)):
                            score += 0.6
                        for key, syn_list in self.default_mappings.items():
                            if mapped_term.lower() == key or mapped_term.lower() in [s.lower() for s in syn_list]:
                                if table_name.lower() == key:
                                    score += 0.7
                        score += schema_boost
                        # Embedding similarity
                        if score < 1.0:
                            try:
                                table_embedding = self._encode_query(table_name)[0]
                                token_embedding = self._encode_query(mapped_term)[0]
                                sim = np.dot(token_embedding, table_embedding) / (
                                    np.linalg.norm(token_embedding) * np.linalg.norm(table_embedding)
                                )
                                score += max(0.0, min(0.2, float(sim)))
                            except TIAError as e:
                                self.logger.warning(f"Embedding failed for token {mapped_term}: {str(e)}")
                        if score > 0:
                            table_scores[full_table] = table_scores.get(full_table, 0.0) + score
                            self.logger.debug(f"Table {full_table} scored {score} for token {mapped_term}")
                        for column in table.get("columns", []):
                            if not isinstance(column, dict) or "name" not in column:
                                self.logger.warning(f"Invalid column entry in table {table_name}: {column}")
                                continue
                            col_name = column["name"]
                            col_synonyms = synonyms[schema].get(f"{table_name}.{col_name}", [])
                            col_score = 0.0
                            if mapped_term.lower() == col_name.lower():
                                col_score += 1.0
                            elif (mapped_term.lower() in col_name.lower() or
                                  any(mapped_term.lower() in s.lower() for s in col_synonyms)):
                                col_score += 0.8
                            if col_score > 0:
                                full_col = f"{schema}.{table_name}.{col_name}"
                                if full_col not in result["columns"]:
                                    result["columns"].append(full_col)
                                    table_scores[full_table] = table_scores.get(full_table, 0.0) + (col_score * 0.5)
                            if "unique_values" in column:
                                for value in column["unique_values"]:
                                    if token.lower() == value.lower():
                                        result["extracted_values"][f"{table_name}.{col_name}"] = value
                                        if "?" not in result["placeholders"]:
                                            result["placeholders"].append("?")
            self.logger.debug(f"Token mappings for NLQ '{nlq}': {'; '.join(token_mappings)}")
            # Select top tables
            selected_tables = []
            for full_table, score in sorted(table_scores.items(), key=lambda x: x[1], reverse=True)[:3]:
                if score >= 0.9:
                    schema, table_name = full_table.split(".")
                    if any(t.get("name") == table_name for t in metadata.get(schema, {}).get("tables", [])):
                        selected_tables.append(full_table)
            result["tables"] = selected_tables
            if not result["tables"]:
                self.logger.debug(f"No tables scored >= 0.9 for NLQ: {nlq}")
                return None
            # Handle references
            for schema in schemas:
                schema_data = metadata.get(schema, {})
                for table in schema_data.get("tables", []):
                    if not isinstance(table, dict) or "name" not in table:
                        self.logger.warning(f"Invalid table entry in metadata: {table}")
                        continue
                    table_name = table.get("name")
                    full_table = f"{schema}.{table_name}"
                    for column in table.get("columns", []):
                        if not isinstance(column, dict) or "name" not in column:
                            self.logger.warning(f"Invalid column entry in table {table_name}: {column}")
                            continue
                        if column.get("references"):
                            ref_table = column["references"]["table"]
                            ref_schema = schema if "." not in ref_table else ref_table.split(".")[0]
                            ref_table_name = ref_table.split(".")[1] if "." in ref_table else ref_table
                            full_ref_table = f"{ref_schema}.{ref_table_name}"
                            if (full_ref_table in selected_tables and full_table not in selected_tables and
                                any(ref_table_name.lower() in t.lower() for t in tokens + list(synonyms[schema].get(ref_table_name, [])))):
                                selected_tables.append(full_table)
            result["tables"] = selected_tables
            if not result["tables"]:
                self.logger.debug(f"No tables identified for NLQ: {nlq} in schemas {schemas}")
                return None
            result["ddl"] = self._generate_ddl([t.split(".")[-1] for t in result["tables"]], schemas[0])
            self.logger.debug(f"Generated metadata-based identification for NLQ: {nlq}, tables: {result['tables']}")
            if self.enable_component_logging:
                print(f"Component Output: Metadata-based prediction identified tables {result['tables']} for NLQ '{nlq}'")
            return result
        except TIAError as e:
            self.logger.error(f"Metadata identification error for NLQ '{nlq}': {str(e)}\n{traceback.format_exc()}")
            return None

    def _fallback_prediction(self, nlq: str, schemas: List[str], nlp_processor: NLPProcessor) -> Dict:
        """Fallback prediction for common NLQs.

        Args:
            nlq (str): Natural language query.
            schemas (List[str]): Schema names.
            nlp_processor (NLPProcessor): NLP processor instance.

        Returns:
            Dict: Fallback identification result.
        """
        try:
            nlp_result = nlp_processor.process_query(nlq, schemas[0], datasource=self.datasource)
            tokens = nlp_result.get("tokens", [])
            entities = nlp_result.get("entities", {})
            extracted_values = nlp_result.get("extracted_values", {})
            metadata = self._get_metadata(schemas)
            result = {
                "tables": [],
                "columns": [],
                "extracted_values": extracted_values,
                "placeholders": ["?" for _ in extracted_values],
                "entities": entities,
                "ddl": "",
                "conditions": self._extract_conditions(nlp_result),
                "sql": None
            }
            for token in tokens:
                for key, syn_list in self.default_mappings.items():
                    if token.lower() == key or token.lower() in [s.lower() for s in syn_list]:
                        for schema in schemas:
                            schema_data = metadata.get(schema, {})
                            for table in schema_data.get("tables", []):
                                if not isinstance(table, dict) or "name" not in table:
                                    continue
                                if table.get("name").lower() == key:
                                    full_table = f"{schema}.{table['name']}"
                                    if full_table not in result["tables"]:
                                        result["tables"].append(full_table)
            if result["tables"]:
                result["ddl"] = self._generate_ddl([t.split(".")[-1] for t in result["tables"]], schemas[0])
                self.logger.debug(f"Fallback prediction for NLQ '{nlq}': {result['tables']}")
                if self.enable_component_logging:
                    print(f"Component Output: Fallback prediction identified tables {result['tables']} for NLQ '{nlq}'")
            return result
        except Exception as e:
            self.logger.error(f"Fallback prediction failed for NLQ '{nlq}': {str(e)}\n{traceback.format_exc()}")
            return {
                "tables": [],
                "columns": [],
                "extracted_values": {},
                "placeholders": [],
                "entities": {},
                "ddl": "",
                "conditions": {},
                "sql": None
            }

    def _encode_query(self, text: str | List[str]) -> np.ndarray:
        """Encode text using Azure Open AI embeddings with LRU cache.

        Args:
            text (str | List[str]): Text to encode.

        Returns:
            np.ndarray: Embeddings.

        Raises:
            TIAError: If encoding fails.
        """
        try:
            azure_config = self.config_utils.load_azure_config()
            required_keys = ["api_key", "azure_endpoint", "embedding_deployment_name"]
            missing_keys = [k for k in required_keys if k not in azure_config]
            if missing_keys:
                self.logger.error(f"Missing Azure configuration keys: {missing_keys}")
                raise TIAError(f"Missing Azure configuration keys: {missing_keys}")
            self.embedding_deployment_name = azure_config.get("embedding_deployment_name", self.embedding_deployment_name)
            if self.embedding_deployment_name == self.model_name:
                self.logger.warning(
                    f"Embedding deployment name ({self.embedding_deployment_name}) matches model name, "
                    f"likely incorrect. Expected 'embedding-model'."
                )
            api_version = azure_config.get("api_version", "2024-12-01-preview")
            custom_auth_headers = azure_config.get("custom_auth_headers", {})
            self.logger.debug(
                f"Encoding query with azure_endpoint={azure_config['azure_endpoint']}, "
                f"deployment={self.embedding_deployment_name}, api_version={api_version}, "
                f"input={text[:100]}..."
            )
            client = AzureOpenAI(
                api_key=azure_config["api_key"],
                api_version=api_version,
                azure_endpoint=azure_config["azure_endpoint"],
                http_client=httpx.Client()
            )
            if isinstance(text, str):
                text = [text]
            cached_embeddings = []
            uncached_texts = []
            for t in text:
                if t in self._embedding_cache:
                    cached_embeddings.append(self._embedding_cache[t])
                    self._embedding_cache.move_to_end(t)  # Update LRU order
                else:
                    uncached_texts.append(t)
            if uncached_texts:
                self.logger.debug(f"Calling embeddings.create with model={self.embedding_deployment_name} for {len(uncached_texts)} texts")
                response = client.embeddings.create(
                    input=uncached_texts,
                    model=self.embedding_deployment_name,
                    extra_headers=custom_auth_headers
                )
                new_embeddings = [data.embedding for data in response.data]
                for t, emb in zip(uncached_texts, new_embeddings):
                    self._embedding_cache[t] = emb
                    if len(self._embedding_cache) > self._embedding_cache_max_size:
                        self._embedding_cache.popitem(last=False)  # Remove least recently used
                cached_embeddings.extend(new_embeddings)
            embeddings = np.array(cached_embeddings)
            self.logger.debug(f"Encoded {len(text)} queries using deployment {self.embedding_deployment_name}")
            if self.enable_component_logging:
                print(f"Component Output: Encoded {len(text)} queries, cache size {len(self._embedding_cache)}")
            return embeddings
        except Exception as e:
            self.logger.error(
                f"Failed to encode query: {str(e)}\n"
                f"Stack trace: {traceback.format_exc()}\n"
                f"Input: {text[:100]}..., Deployment: {self.embedding_deployment_name}"
            )
            raise TIAError(f"Failed to encode query: {str(e)}")

    def _generate_ddl(self, tables: List[str], schema: str) -> str:
        """Generate DDL statement for tables.

        Args:
            tables (List[str]): Table names.
            schema (str): Schema name.

        Returns:
            str: DDL string.
        """
        try:
            metadata = self._get_metadata([schema])
            ddl_parts = []
            for table in tables:
                table_data = next((t for t in metadata.get(schema, {}).get("tables", []) if t.get("name") == table), {})
                if not table_data:
                    self.logger.warning(f"No metadata found for table {table} in schema {schema}")
                    continue
                columns = [f"{col['name']} {col['type']}" for col in sorted(table_data.get("columns", []), key=lambda x: x['name'])]
                if columns:
                    ddl_parts.append(f"CREATE TABLE {schema}.{table} ({', '.join(columns)});")
            ddl = "\n".join(ddl_parts)
            self.logger.debug(f"Generated DDL for tables {tables}")
            if self.enable_component_logging:
                print(f"Component Output: Generated DDL for {len(tables)} tables")
            return ddl
        except TIAError as e:
            self.logger.error(f"Failed to generate DDL for tables {tables}: {str(e)}\n{traceback.format_exc()}")
            return ""

    def _extract_conditions(self, nlp_result: Dict) -> Dict:
        """Extract conditions from NLP result.

        Args:
            nlp_result (Dict): NLP processing result.

        Returns:
            Dict: Conditions dictionary.
        """
        conditions = []
        try:
            for key, value in nlp_result.get("extracted_values", {}).items():
                if isinstance(value, str) and "date" in key.lower():
                    try:
                        year = int(value)
                        conditions.append(f"EXTRACT(YEAR FROM {key}) = {year}")
                    except ValueError:
                        conditions.append(f"{key} = '{value}'")
                elif isinstance(value, list):
                    conditions.append(f"{key} IN {tuple(value)}")
                else:
                    conditions.append(f"{key} = '{value}'")
            result = {"conditions": conditions}
            self.logger.debug(f"Extracted {len(conditions)} conditions")
            if self.enable_component_logging:
                print(f"Component Output: Extracted {len(conditions)} conditions")
            return result
        except Exception as e:
            self.logger.error(f"Failed to extract conditions: {str(e)}\n{traceback.format_exc()}")
            return {"conditions": []}

    def _get_stored_sql(self, query: str) -> Optional[str]:
        """Retrieve stored SQL from training data.

        Args:
            query (str): NLQ to match.

        Returns:
            Optional[str]: SQL query or None.
        """
        try:
            db_manager = DBManager(self.config_utils)
            training_data = db_manager.get_training_data(self.datasource)
            for row in training_data:
                if row["user_query"] == query:
                    self.logger.debug(f"Found stored SQL for query '{query}'")
                    return row["relevant_sql"]
            return None
        except Exception as e:
            self.logger.error(f"Failed to retrieve stored SQL: {str(e)}\n{traceback.format_exc()}")
            return None

    def train_manual(self, datasource: Dict, nlq: str, tables: List[str], columns: List[str], extracted_values: Dict, placeholders: List[str], sql: str) -> None:
        """Store manual training data.

        Args:
            datasource (Dict): Datasource configuration.
            nlq (str): Natural language query.
            tables (List[str]): Related tables.
            columns (List[str]): Specific columns.
            extracted_values (Dict): Extracted values.
            placeholders (List[str]): Placeholders.
            sql (str): SQL query.

        Raises:
            TIAError: If storage fails.
        """
        self._set_datasource(datasource)
        if not all([nlq, tables, columns, sql]):
            self.logger.error("Missing required training data fields")
            raise TIAError("Missing required training data fields")
        training_data = {
            "db_source_type": self.datasource["type"],
            "db_name": self.datasource["name"],
            "user_query": nlq,
            "related_tables": json.dumps(tables),
            "specific_columns": json.dumps(columns),
            "extracted_values": json.dumps(extracted_values),
            "placeholders": json.dumps(placeholders),
            "relevant_sql": sql,
            "llm_sql": sql,
            "is_lsql_valid": True,
            "context_text1": nlq,
            "context_text2": "",
            "IS_SLM_TRAINED": False,
            "SCENARIO_ID": self._get_next_scenario_id()
        }
        try:
            db_manager = DBManager(self.config_utils)
            db_manager.store_training_data(self.datasource, [training_data])
            self.logger.info(f"Stored manual training data for NLQ: {nlq}, scenario_id: {training_data['SCENARIO_ID']}")
            if self.enable_component_logging:
                print(f"Component Output: Stored manual training data for NLQ '{nlq}'")
        except Exception as e:
            self.logger.error(f"Failed to store training data: {str(e)}\n{traceback.format_exc()}")
            raise TIAError(f"Failed to store training data: {str(e)}")

    def train_bulk(self, datasource: Dict, training_data: List[Dict]) -> None:
        """Store bulk training data.

        Args:
            datasource (Dict): Datasource configuration.
            training_data (List[Dict]): List of training data dictionaries.

        Raises:
            TIAError: If storage fails.
        """
        self._set_datasource(datasource)
        try:
            processed_data = []
            for data in training_data[:100]:
                if not all(key in data for key in ["user_query", "related_tables", "specific_columns", "relevant_sql"]):
                    self.logger.warning(f"Invalid bulk training data entry: {data}")
                    continue
                data["db_source_type"] = self.datasource["type"]
                data["db_name"] = self.datasource["name"]
                data["extracted_values"] = json.dumps(data.get("extracted_values", {}))
                data["placeholders"] = json.dumps(data.get("placeholders", []))
                data["llm_sql"] = data["relevant_sql"]
                data["is_lsql_valid"] = True
                data["context_text1"] = data["user_query"]
                data["context_text2"] = ""
                data["IS_SLM_TRAINED"] = False
                data["SCENARIO_ID"] = self._get_next_scenario_id()
                processed_data.append(data)
            if processed_data:
                db_manager = DBManager(self.config_utils)
                db_manager.store_training_data(self.datasource, processed_data)
                self.logger.info(f"Stored {len(processed_data)} bulk training records")
                if self.enable_component_logging:
                    print(f"Component Output: Stored {len(processed_data)} bulk training records")
        except Exception as e:
            self.logger.error(f"Failed to store bulk training data: {str(e)}\n{traceback.format_exc()}")
            raise TIAError(f"Failed to store bulk training data: {str(e)}")

    def _get_next_scenario_id(self) -> str:
        """Get the next SCENARIO_ID for training data as a string.

        Returns:
            str: Next SCENARIO_ID (e.g., 'SCN_000001').
        """
        try:
            db_manager = DBManager(self.config_utils)
            training_data = db_manager.get_training_data(self.datasource)
            scenario_ids = [row["SCENARIO_ID"] for row in training_data if row.get("SCENARIO_ID")]
            max_numeric = 0
            for sid in scenario_ids:
                match = re.match(r"SCN_(\d+)", str(sid))
                if match:
                    max_numeric = max(max_numeric, int(match.group(1)))
            next_numeric = max_numeric + 1
            next_id = f"SCN_{next_numeric:06d}"
            self.logger.debug(f"Generated scenario_id: {next_id}")
            if self.enable_component_logging:
                print(f"Component Output: Generated scenario_id {next_id}")
            return next_id
        except Exception as e:
            self.logger.error(f"Failed to retrieve scenario ID: {str(e)}\n{traceback.format_exc()}")
            return "SCN_000001"

    def train(self, datasource: Dict, training_data: List[Dict]) -> None:
        """Train the prediction model using provided training data.

        Args:
            datasource (Dict): Datasource configuration.
            training_data (List[Dict]): List of training data dictionaries.

        Raises:
            TIAError: If training fails.
        """
        self._set_datasource(datasource)
        if not self.model_path.exists():
            self.logger.info(f"No model found at {self.model_path}, generating default")
            self.generate_model(datasource)
        try:
            queries = [data["user_query"] for data in training_data]
            tables = [json.loads(data["related_tables"]) for data in training_data]
            columns = [json.loads(data["specific_columns"]) for data in training_data]
            embeddings = self._encode_query(queries)
            model_data = {
                "queries": queries,
                "tables": tables,
                "columns": columns,
                "embeddings": embeddings.tolist()
            }
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.model_path, "w") as f:
                json.dump(model_data, f, indent=2)
            self.logger.info(f"Trained model and saved at {self.model_path}")
            if self.enable_component_logging:
                print(f"Component Output: Trained model with {len(queries)} queries, saved at {self.model_path}")
            metrics = {
                "model_version": datetime.now().strftime("%Y%m%d%H%M%S"),
                "precision": 0.095,
                "recall": 0.0,
                "nlq_breakdown": {q: {"precision": [], "recall": []} for q in queries}
            }
            db_manager = DBManager(self.config_utils)
            db_manager.store_model_metrics(self.datasource, metrics)
            self._load_model()
        except Exception as e:
            self.logger.error(f"Failed to train model: {str(e)}\n{traceback.format_exc()}")
            raise TIAError(f"Failed to train model: {str(e)}")

    def generate_model(self, datasource: Dict, force: bool = False) -> None:
        """Generate a default prediction model.

        Args:
            datasource (Dict): Datasource configuration.
            force (bool): If True, overwrite existing model.

        Raises:
            TIAError: If model generation fails.
        """
        if self.generating_model:
            self.logger.warning(f"Model generation already in progress for {datasource['name']}, skipping")
            return
        self.generating_model = True
        try:
            required_keys = ["name", "type", "connection"]
            if not all(key in datasource for key in required_keys):
                self.logger.error("Missing required keys in datasource configuration")
                raise TIAError("Missing required keys")
            self.datasource = datasource
            self.model_path = self.config_utils.models_dir / f"model_{datasource['name']}.json"
            if self.model_path.exists() and not force:
                self.logger.debug(f"Model already exists at {self.model_path}, skipping generation")
                self._load_model()
                return
            self.logger.info(f"Generating default model for {datasource['name']}, force={force}")
            # Initialize empty model to support incremental training
            model_data = {
                "queries": [],
                "tables": [],
                "columns": [],
                "embeddings": []
            }
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.model_path, "w") as f:
                json.dump(model_data, f, indent=2)
            self.logger.info(f"Generated default model at {self.model_path}")
            if self.enable_component_logging:
                print(f"Component Output: Generated default model at {self.model_path}")
            try:
                db_manager = DBManager(self.config_utils)
                metrics = {
                    "model_version": datetime.now().strftime("%Y%m%d%H%M%S"),
                    "precision": 0.0,
                    "recall": 0.0,
                    "nlq_breakdown": {}
                }
                db_manager.store_model_metrics(self.datasource, metrics)
            except Exception as db_e:
                self.logger.error(f"Failed to store model metrics: {str(db_e)}\n{traceback.format_exc()}")
            self._load_model()
        except Exception as e:
            self.logger.error(f"Failed to generate default model: {str(e)}\n{traceback.format_exc()}")
            raise TIAError(f"Failed to generate default model: {str(e)}")
        finally:
            self.generating_model = False

    def regenerate_model(self, datasource: Dict) -> None:
        """Force regeneration of the prediction model.

        Args:
            datasource (Dict): Datasource configuration.

        Raises:
            TIAError: If regeneration fails.
        """
        try:
            self.logger.info(f"Forcing model regeneration for {datasource['name']}")
            if self.enable_component_logging:
                print(f"Component Output: Forcing model regeneration for {datasource['name']}")
            self.generate_model(datasource, force=True)
        except TIAError as e:
            self.logger.error(f"Failed to regenerate model: {str(e)}\n{traceback.format_exc()}")
            raise
