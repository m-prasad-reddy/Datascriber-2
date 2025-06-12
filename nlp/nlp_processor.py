import spacy
import logging
import json
import re
from typing import Dict, List, Optional, Set
from config.utils import ConfigUtils, ConfigError
from datetime import datetime
from pathlib import Path

class NLPError(Exception):
    """Custom exception for NLP processing errors."""
    pass

class NLPProcessor:
    """NLP processor for extracting tokens and entities from NLQs.

    Uses spaCy for tokenization and entity recognition, with synonym mapping and
    regex-based entity extraction for table/column identification in the Datascriber project.

    Attributes:
        config_utils (ConfigUtils): Configuration utility instance.
        logger (logging.Logger): System-wide logger.
        nlp (spacy.language.Language): spaCy NLP model.
        date_patterns (List[re.Pattern]): Regex patterns for date extraction.
        default_mappings (Dict): Default synonym mappings from config.
    """

    def __init__(self, config_utils: ConfigUtils, logger: logging.Logger):
        """Initialize NLPProcessor.

        Args:
            config_utils (ConfigUtils): Configuration utility instance.
            logger (logging.Logger): System logger.

        Raises:
            NLPError: If initialization fails.
        """
        self.config_utils = config_utils
        self.logger = logger
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["parser"])
            custom_stop_words = {"the", "a", "an"}  # Preserve 'all', 'list'
            for word in custom_stop_words:
                if word in self.nlp.Defaults.stop_words:
                    self.nlp.Defaults.stop_words.remove(word)
            self.date_patterns = [
                re.compile(r"\b\d{4}\b"),  # YYYY
                re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),  # MM/DD/YYYY
                re.compile(r"\b\d{1,2}-\d{1,2}-\d{2,4}\b"),  # MM-DD-YYYY
            ]
            self.default_mappings = self._load_default_mappings()
            self.logger.debug("Initialized NLPProcessor with spaCy model en_core_web_sm")
        except Exception as e:
            self.logger.error(f"Failed to initialize NLPProcessor: {str(e)}")
            raise NLPError(f"Failed to initialize NLPProcessor: {str(e)}")

    def _load_default_mappings(self) -> Dict:
        """Load default synonym mappings from default_mappings.json.

        Returns:
            Dict: Default mappings.

        Raises:
            NLPError: If loading fails critically.
        """
        try:
            config_path = self.config_utils.config_dir / "default_mappings.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = json.load(f)
                self.logger.debug(f"Loaded default mappings from {config_path}")
                return config.get("common_mappings", {})
            self.logger.warning(f"default_mappings.json not found at {config_path}, using fallback. Create file with 'common_mappings' for scalability.")
            default = {
                "customers": ["customer", "clients", "users"],
                "orders": ["order", "purchases"],
                "products": ["product", "items"],
                "stores": ["store", "shops"],
                "staffs": ["staff", "employees"]
            }
            return default
        except (json.JSONDecodeError, FileNotFoundError) as e:
            self.logger.error(f"Failed to load default_mappings.json: {str(e)}, using fallback")
            return {
                "customers": ["customer", "clients", "users"],
                "orders": ["order", "purchases"],
                "products": ["product", "items"],
                "stores": ["store", "shops"],
                "staffs": ["staff", "employees"]
            }

    def process_query(self, query: str, schema: str, datasource: Optional[Dict] = None) -> Dict:
        """Process an NLQ to extract tokens, entities, and values.

        Args:
            query (str): Natural language query.
            schema (str): Schema name.
            datasource (Optional[Dict]): Datasource configuration.

        Returns:
            Dict: Processed query with tokens, entities, and extracted values.

        Raises:
            NLPError: If processing fails.
        """
        try:
            if not query or not isinstance(query, str):
                self.logger.error("Invalid query provided")
                raise NLPError("Invalid query provided")
            if not schema or not isinstance(schema, str):
                self.logger.error("Invalid schema provided")
                raise NLPError("Invalid schema provided")
            if datasource and not isinstance(datasource, dict):
                self.logger.error("Invalid datasource provided")
                raise NLPError("Invalid datasource provided")
            nlp_query = query.strip()
            tokens = self._tokenize(nlp_query)
            entities = self._extract_entities(nlp_query)
            # Fallback for weak tokenization
            if not tokens or len(tokens) < 2:
                self.logger.warning(f"Insufficient tokens extracted for query: {nlp_query}, using fallback")
                tokens = [word for word in nlp_query.lower().split() if word]
            # Ensure key terms from default mappings
            for key in self.default_mappings:
                if key in nlp_query.lower() and key not in tokens:
                    tokens.append(key)
            # Ensure entities for key terms
            if not entities.get("objects"):
                for key in self.default_mappings:
                    if key in nlp_query.lower():
                        entities["objects"].append(key)
            synonyms = {}
            if datasource:
                try:
                    synonyms = self._load_synonyms(datasource, schema)
                    tokens = [self.map_synonyms(token, synonyms, schema, datasource) for token in tokens]
                except NLPError as e:
                    self.logger.warning(f"Failed to load synonyms for schema {schema}: {str(e)}")
            result = {
                "tokens": [token for token in tokens if token],
                "entities": entities,
                "extracted_values": self._extract_values(entities)
            }
            self.log_processing_metrics(nlp_query, result)
            self.logger.debug(f"Processed query: {nlp_query}, tokens: {result['tokens']}, entities: {result['entities']}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to process query '{nlp_query}' for schema {schema}: {str(e)}")
            raise NLPError(f"Failed to process query: {str(e)}")

    def _tokenize(self, query: str) -> List[str]:
        """Tokenize the query using spaCy with fallback.

        Args:
            query (str): Query to tokenize.

        Returns:
            List[str]: List of tokens.
        """
        try:
            doc = self.nlp(query.lower())
            tokens = [
                token.text for token in doc
                if not token.is_stop and not token.is_punct and token.text.strip()
            ]
            if len(tokens) < 2:
                self.logger.warning(f"Low token count for query '{query}', using fallback")
                tokens = [word.strip() for word in query.lower().split() if word.strip()]
            self.logger.debug(f"Tokenized query '{query}' to: {tokens}")
            return tokens
        except Exception as e:
            self.logger.error(f"Failed to tokenize query '{query}': {str(e)}")
            return [word.strip() for word in query.lower().split() if word.strip()]

    def _extract_entities(self, query: str) -> Dict:
        """Extract entities (dates, names, objects, places) from query.

        Args:
            query (str): Query to process.

        Returns:
            Dict: Dictionary of entities.
        """
        entities = {"common": [], "dates": [], "names": [], "objects": [], "places": []}
        try:
            doc = self.nlp(query)
            for ent in doc.ents:
                if ent.label_ == "DATE":
                    entities["dates"].append(ent.text)
                elif ent.label_ == "PERSON":
                    entities["names"].append(ent.text)
                elif ent.label_ == "GPE":
                    entities["places"].append(ent.text)
                elif ent.label_ in ["PRODUCT", "ORG"]:
                    entities["objects"].append(ent.text)
            # Rule-based extraction using default mappings
            for key in self.default_mappings:
                if key in query.lower():
                    entities["objects"].append(key)
            for pattern in self.date_patterns:
                for match in pattern.finditer(query):
                    entities["dates"].append(match.group(0))
            # Remove duplicates
            for key in entities:
                entities[key] = list(dict.fromkeys(entities[key]))
            if not any(entities.values()):
                self.logger.warning(f"No entities extracted for query: {query}")
            self.logger.debug(f"Extracted entities for query '{query}': {entities}")
            return entities
        except Exception as e:
            self.logger.error(f"Failed to extract entities for query '{query}': {str(e)}")
            return entities

    def _extract_values(self, entities: Dict) -> Dict:
        """Extract values from entities for query parameterization.

        Args:
            entities (Dict): Dictionary of entities.

        Returns:
            Dict: Extracted values.
        """
        extracted_values = {}
        try:
            for key, values in entities.items():
                for value in values:
                    if key == "dates":
                        try:
                            year = int(value)
                            extracted_values[f"{key}_year"] = str(year)
                        except ValueError:
                            extracted_values[key] = value
                    else:
                        extracted_values[key] = value
            self.logger.debug(f"Extracted values: {extracted_values}")
            return extracted_values
        except Exception as e:
            self.logger.error(f"Failed to extract values from entities: {str(e)}")
            return extracted_values

    def _load_synonyms(self, datasource: Dict, schema: str) -> Dict[str, List[str]]:
        """Load synonyms for schema and datasource.

        Args:
            datasource (Dict): Datasource configuration.
            schema (str): Schema name.

        Returns:
            Dict[str, List[str]]: Synonym mappings.

        Raises:
            NLPError: If synonym loading fails.
        """
        synonyms = {}
        try:
            config_path = self.config_utils.config_dir / "synonym_config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = json.load(f)
                synonyms.update(config.get("synonyms", {}))
            metadata = self.config_utils.load_metadata(datasource["name"], [schema]).get(schema, {})
            for table in metadata.get("tables", {}).values():
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
            self.logger.debug(f"Loaded synonyms for schema {schema}: {synonyms}")
            return synonyms
        except (json.JSONDecodeError, ConfigError) as e:
            self.logger.error(f"Failed to load synonyms for schema {schema}: {str(e)}")
            raise NLPError(f"Failed to load synonyms: {str(e)}")

    def map_synonyms(self, token: str, synonyms: Dict[str, List[str]], schema: str, datasource: Dict) -> str:
        """Map token to schema element using synonyms.

        Args:
            token (str): Token to map.
            synonyms (Dict[str, List[str]]): Synonym mappings.
            schema (str): Schema name.
            datasource (Dict): Datasource configuration.

        Returns:
            str: Mapped token or original token.
        """
        try:
            token_lower = token.lower()
            for key, syn_list in synonyms.items():
                key_lower = key.lower()
                if token_lower == key_lower or token_lower in [s.lower() for s in syn_list]:
                    # Validate schema
                    if "." in key_lower:
                        key_schema = key_lower.split(".")[0]
                        if key_schema != schema.lower():
                            continue
                    self.logger.debug(f"Mapped token '{token}' to '{key}'")
                    return key
            # Check default mappings
            for mapped_term, syn_list in self.default_mappings.items():
                if token_lower == mapped_term or token_lower in [s.lower() for s in syn_list]:
                    # Find matching table in schema
                    metadata = self.config_utils.load_metadata(datasource["name"], [schema]).get(schema, {})
                    for table in metadata.get("tables", {}).values():
                        if table.get("name").lower() == mapped_term:
                            mapped_key = f"{schema}.{table['name']}"
                            self.logger.debug(f"Mapped token '{token}' to '{mapped_key}' via default mappings")
                            return mapped_key
            self.logger.debug(f"No synonym mapping found for token '{token}'")
            return token
        except Exception as e:
            self.logger.error(f"Failed to map synonyms for token '{token}': {str(e)}")
            return token

    def validate_entities(self, entities: Dict, schema: str, datasource: Dict) -> Dict:
        """Validate entities against schema metadata.

        Args:
            entities (Dict): Extracted entities.
            schema (str): Schema name.
            datasource (Dict): Datasource configuration.

        Returns:
            Dict: Validated entities.
        """
        validated_entities = {"common": [], "dates": [], "names": [], "objects": [], "places": []}
        try:
            metadata = self.config_utils.load_metadata(datasource["name"], [schema]).get(schema, {})
            tables = [table["name"].lower() for table in metadata.get("tables", {}).values()]
            for key, values in entities.items():
                for value in values:
                    if key == "objects" and value.lower() in tables:
                        validated_entities[key].append(value)
                    elif key in ["common", "dates", "names", "places"]:
                        validated_entities[key].append(value)
            # Validate against default mappings
            for key in self.default_mappings:
                if key in entities.get("objects", []) and key in tables:
                    validated_entities["objects"].append(key)
            self.logger.debug(f"Validated entities for schema {schema}: {validated_entities}")
            return validated_entities
        except ConfigError as e:
            self.logger.error(f"Failed to validate entities for schema {schema}: {str(e)}")
            return entities

    def extract_regex_entities(self, query: str) -> Dict:
        """Extract entities using regex patterns.

        Args:
            query (str): Query to process.

        Returns:
            Dict: Extracted entities.
        """
        entities = {"common": [], "dates": [], "names": [], "objects": [], "places": []}
        try:
            for pattern in self.date_patterns:
                for match in pattern.finditer(query):
                    entities["dates"].append(match.group(0))
            name_pattern = re.compile(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b")
            for match in name_pattern.finditer(query):
                entities["names"].append(match.group(0))
            # Rule-based object detection
            for key in self.default_mappings:
                if key in query.lower():
                    entities["objects"].append(key)
            self.logger.debug(f"Extracted regex entities for query '{query}': {entities}")
            return entities
        except Exception as e:
            self.logger.error(f"Failed to extract regex entities for query '{query}': {str(e)}")
            return entities

    def merge_entities(self, spacy_entities: Dict, regex_entities: Dict) -> Dict:
        """Merge spaCy and regex-based entities.

        Args:
            spacy_entities (Dict): Entities from spaCy.
            regex_entities (Dict): Entities from regex.

        Returns:
            Dict: Merged entities.
        """
        merged = {"common": [], "dates": [], "names": [], "objects": [], "places": []}
        try:
            for key in merged:
                merged[key] = list(set(spacy_entities.get(key, []) + regex_entities.get(key, [])))
            self.logger.debug(f"Merged entities: {merged}")
            return merged
        except Exception as e:
            self.logger.error(f"Failed to merge entities: {str(e)}")
            return spacy_entities

    def process_dynamic_synonyms(self, query: str, schema: str, datasource: Dict) -> Dict:
        """Process dynamic synonyms based on query context.

        Args:
            query (str): Natural language query.
            schema (str): Schema name.
            datasource (Dict): Datasource configuration.

        Returns:
            Dict: Dynamic synonym mappings.
        """
        dynamic_synonyms = {}
        try:
            metadata = self.config_utils.load_metadata(datasource["name"], [schema]).get(schema, {})
            tokens = self._tokenize(query)
            for table in metadata.get("tables", {}).values():
                if not isinstance(table, dict) or "name" not in table:
                    continue
                table_name = table.get("name").lower()
                for token in tokens:
                    if token.lower() in table_name or table_name in token.lower():
                        dynamic_synonyms[token] = [table_name]
            # Add default mappings
            for key, syn_list in self.default_mappings.items():
                if key in query.lower():
                    dynamic_synonyms[key] = syn_list
            self.logger.debug(f"Processed dynamic synonyms for query '{query}': {dynamic_synonyms}")
            return dynamic_synonyms
        except ConfigError as e:
            self.logger.error(f"Failed to process dynamic synonyms for schema {schema}: {str(e)}")
            return dynamic_synonyms

    def validate_query_context(self, query: str, schema: str, datasource: Dict) -> bool:
        """Validate query context against schema and datasource.

        Args:
            query (str): Natural language query.
            schema (str): Schema name.
            datasource (Dict): Datasource configuration.

        Returns:
            bool: True if context is valid, False otherwise.
        """
        try:
            metadata = self.config_utils.load_metadata(datasource["name"], [schema]).get(schema, {})
            tokens = self._tokenize(query)
            table_names = [table["name"].lower() for table in metadata.get("tables", {}).values()]
            for token in tokens:
                if token.lower() in table_names:
                    return True
            for key in self.default_mappings:
                if key in query.lower() and key in table_names:
                    return True
            self.logger.warning(f"No valid context found for query '{query}' in schema {schema}")
            return False
        except ConfigError as e:
            self.logger.error(f"Failed to validate query context for schema {schema}: {str(e)}")
            return False

    def enhance_query(self, query: str, schema: str, datasource: Dict) -> str:
        """Enhance query with contextual information.

        Args:
            query (str): Natural language query.
            schema (str): Schema name.
            datasource (Dict): Datasource configuration.

        Returns:
            str: Enhanced query.
        """
        try:
            enhanced_query = query
            if self.validate_query_context(query, schema, datasource):
                entities = self._extract_entities(query)
                if entities.get("objects"):
                    enhanced_query += f" (objects: {', '.join(entities['objects'])})"
                if entities.get("dates"):
                    enhanced_query += f" (dates: {', '.join(entities['dates'])})"
            self.logger.debug(f"Enhanced query: {enhanced_query}")
            return enhanced_query
        except Exception as e:
            self.logger.error(f"Failed to enhance query '{query}': {str(e)}")
            return query

    def log_processing_metrics(self, query: str, result: Dict) -> None:
        """Log processing metrics for the query.

        Args:
            query (str): Natural language query.
            result (Dict): Processing result with tokens and entities.
        """
        try:
            metrics = {
                "query_length": len(query),
                "token_count": len(result.get("tokens", [])),
                "entity_count": sum(len(values) for values in result.get("entities", {}).values()),
                "timestamp": datetime.now().isoformat()
            }
            self.logger.info(f"Processing metrics for query '{query}': {metrics}")
        except Exception as e:
            self.logger.error(f"Failed to log processing metrics for query '{query}': {str(e)}")