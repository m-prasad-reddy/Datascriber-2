# Datascriber Requirements Document

## Purpose
This document outlines the technical and functional requirements for the Datascriber tool, a natural language query (NLQ) processing system that converts user queries into SQL for execution against SQL Server or S3-compatible object storage datasources. The requirements cover all user roles, step-by-step processes, scenarios, example queries, and validation rules derived from development discussions. The document is intended as input for designing the application using organization-specific tools, procedures, and cloud service providers, ensuring compatibility with S3-protocol-enabled object storage and Azure Open AI services with token-based authentication.

## Scope
Datascriber supports:
- Natural language queries for data retrieval (e.g., "Show orders from 2016-01-01").
- S3-compatible object storage (e.g., CSV files in `s3://bike-stores-bucket/data-files/`).
- SQL Server databases.
- Schema validation, metadata-driven table loading, and dynamic date handling.
- User roles: Admin and End User.
- CLI-based interaction with query mode.

## User Roles
1. **Admin**:
   - Manages datasource selection and configuration validation.
   - Executes NLQs for testing and validation.
   - Accesses logs and rejected queries for troubleshooting.
2. **End User**:
   - Submits NLQs in query mode to retrieve data.
   - Receives results or user-friendly error messages.

## Functional Requirements

### FR1: User Authentication
**Description**: Users must authenticate to access Datascriber’s CLI.
- **Steps**:
  1. Launch CLI: `python main.py`.
  2. Enter command: `login <username>`.
  3. System validates username against configured user list.
  4. On success, display: "User <username> logged in."
  5. On failure, display: "Invalid username."
- **Scenarios**:
  - **Scenario 1**: Admin logs in with `login admin`.
    - **Example**: `login admin` → "User admin logged in."
    - **Outcome**: Session initialized, user can select datasource.
  - **Scenario 2**: Invalid username.
    - **Example**: `login invalid_user` → "Invalid username."
    - **Outcome**: Access denied, prompt for retry.
- **Validation Rules**:
  - Username must be a non-empty string.
  - Username must exist in configuration (e.g., `app-config/users.json`).
  - Log successful logins (e.g., `INFO - core.orchestrator - User admin logged in`).
  - Log failed attempts with reason (e.g., `WARNING - core.orchestrator - Invalid username: invalid_user`).

### FR2: Datasource Selection
**Description**: Users select a datasource to query (e.g., `bikestores_s3` or SQL Server).
- **Steps**:
  1. Enter command: `select-datasource <datasource_name>`.
  2. System loads datasource configuration from `app-config/db_configurations.json`.
  3. Validate schemas (e.g., `default` for `bikestores_s3`).
  4. Load metadata (e.g., `data/bikestores_s3/metadata_data_default.json`).
  5. Display: "Selected datasource: <datasource_name>."
- **Scenarios**:
  - **Scenario 1**: Select S3 datasource.
    - **Example**: `select-datasource bikestores_s3` → "Selected datasource: bikestores_s3."
    - **Outcome**: Schemas (`default`) and metadata (11 tables) loaded.
  - **Scenario 2**: Select SQL Server datasource.
    - **Example**: `select-datasource bikestores` → "Selected datasource: bikestores."
    - **Outcome**: Schemas (`sales`, `production`) loaded.
  - **Scenario 3**: Invalid datasource.
    - **Example**: `select-datasource invalid_ds` → "Invalid datasource: invalid_ds."
    - **Outcome**: Prompt for retry, log error.
- **Validation Rules**:
  - Datasource must exist in `db_configurations.json`.
  - Schemas must be non-empty and valid.
  - Metadata file must exist and contain valid JSON with tables and columns.
  - Log schema and metadata loading (e.g., `DEBUG - config.utils - Datasource bikestores_s3: Loaded schemas ['default']`).
  - Store rejected attempts in `rejected_queries` with `error_type="INVALID_DATASOURCE"`.

### FR3: Query Mode Activation
**Description**: Users enter query mode to submit NLQs.
- **Steps**:
  1. Enter command: `query-mode`.
  2. System initializes query mode session.
  3. Prompt: "Enter query (or 'exit' to quit):".
  4. User submits NLQs until exiting with `exit`.
- **Scenarios**:
  - **Scenario 1**: Enter query mode.
    - **Example**: `query-mode` → "Enter query (or 'exit' to quit):".
    - **Outcome**: Ready to process NLQs.
  - **Scenario 2**: User interrupts with Ctrl+C.
    - **Example**: Ctrl+C → "User interrupted query mode."
    - **Outcome**: Exit query mode, return to CLI.
- **Validation Rules**:
  - Query mode requires a selected datasource.
  - Log mode activation (e.g., `DEBUG - cli.interface - Entered query mode`).
  - Log interruptions (e.g., `INFO - cli.interface - User interrupted query mode`).

### FR4: NLQ Processing
**Description**: Process NLQs to generate and execute SQL queries.
- **Steps**:
  1. User submits NLQ (e.g., "Show orders from 2016-01-01").
  2. NLP Processor tokenizes query and extracts entities (e.g., `{'dates': ['2016-01-01'], 'objects': ['orders']}`).
  3. Table Identifier maps tokens to tables using synonyms and embeddings (e.g., `orders` → `default.orders`).
  4. Prompt Generator creates SQL query using Azure Open AI.
  5. Data Executor validates and adjusts SQL (e.g., casts `order_date` to `DATE`).
  6. Execute query against datasource.
  7. Save results to CSV (e.g., `temp/query_results/output_default_*.csv`).
  8. Display sample data (first 5 rows) as table.
- **Scenarios**:
  - **Scenario 1**: Valid date query.
    - **Example**: "Show orders from 2016-01-01" → SQL: `SELECT * FROM orders WHERE order_date = '2016-01-01';`.
    - **Outcome**: CSV with 2 rows, sample data displayed, log: `INFO - cli.interface - Query processed successfully`.
  - **Scenario 2**: Invalid date format.
    - **Example**: "Show orders from 2016" → Error: "Invalid date format '2016'. Please use YYYY-MM-DD (e.g., 2016-01-01)."
    - **Outcome**: No CSV, rejected query stored with `error_type="INVALID_DATE_FORMAT"`.
  - **Scenario 3**: Non-date query.
    - **Example**: "Show customers in NY" → SQL: `SELECT * FROM customers WHERE state = 'NY';`.
    - **Outcome**: CSV with matching rows, sample data displayed.
  - **Scenario 4**: Non-existent table.
    - **Example**: "Show sales" → Error: "No tables identified for query."
    - **Outcome**: No CSV, rejected query stored with `error_type="NO_TABLES"`.
  - **Scenario 5**: Date range query.
    - **Example**: "Show orders between 2016-01-01 and 2016-01-03" → SQL: `SELECT * FROM orders WHERE order_date BETWEEN '2016-01-01' AND '2016-01-03';`.
    - **Outcome**: CSV with matching rows, sample data displayed.
- **Validation Rules**:
  - NLQ must be a non-empty string.
  - Dates must match `YYYY-MM-DD` format (e.g., `2016-01-01`).
  - Tokens must map to valid tables via synonyms or embeddings.
  - SQL query must start with `SELECT` and end with `;`.
  - Results must be non-empty; otherwise, reject with `error_type="NO_DATA"`.
  - Log entity extraction (e.g., `DEBUG - nlp.nlp_processor - Extracted entities: {...}`).
  - Log SQL generation and execution (e.g., `DEBUG - proga.prompt_generator - Generated SQL: ...`).
  - Store rejected queries with error details.

### FR5: Error Handling and Feedback
**Description**: Provide user-friendly error messages and log detailed errors.
- **Steps**:
  1. Detect error (e.g., invalid date, missing table, query failure).
  2. Display user-friendly message in CLI (e.g., "Invalid date format '2016'. Please use YYYY-MM-DD (e.g., 2016-01-01).").
  3. Log detailed error with stack trace to `logs/datascriber.log`.
  4. Store rejected query in `data/<datasource>/datascriber.db`.
- **Scenarios**:
  - **Scenario 1**: Invalid date format.
    - **Example**: "Show orders from 2016" → CLI: "Invalid date format '2016'. Please use YYYY-MM-DD (e.g., 2016-01-01)."
    - **Outcome**: Log: `ERROR - opden.data_executor - Invalid date format '2016'. ...`, rejected query stored.
  - **Scenario 2**: Binder error in DuckDB.
    - **Example**: Misaligned CSV columns → Log: `ERROR - opden.data_executor - DuckDB error loading table orders: Binder Error: ...`.
    - **Outcome**: Fallback loading attempted, rejected query stored with `error_type="DATA_LOAD_ERROR"`.
  - **Scenario 3**: No data returned.
    - **Example**: "Show orders from 2020-01-01" → CLI: "No data returned for query."
    - **Outcome**: Rejected query stored with `error_type="NO_DATA"`.
- **Validation Rules**:
  - CLI messages must be concise and actionable.
  - Logs must include error type, stack trace, and context (e.g., NLQ, schema).
  - Rejected queries must include: `id`, `query`, `timestamp`, `reason`, `user`, `datasource`, `schema`, `error_type`, `user_query`.
  - Error types: `INVALID_DATE_FORMAT`, `DATA_LOAD_ERROR`, `NO_TABLES`, `NO_DATA`, `DUCKDB_ERROR`, `INVALID_SQL`.

### FR6: Result Storage and Display
**Description**: Save query results to CSV and display sample data.
- **Steps**:
  1. Execute query and retrieve results as DataFrame.
  2. Save full results to `temp/query_results/output_<schema>_<timestamp>_<nlq_slug>.csv`.
  3. Display first 5 rows as formatted table in CLI.
  4. Notify user: "Query results saved to: <csv_path>."
- **Scenarios**:
  - **Scenario 1**: Valid query with results.
    - **Example**: "Show orders from 2016-01-01" → CSV: `temp/query_results/output_default_20250617_020608_show_orders_from_2016_01_01.csv`, 2 rows.
    - **Outcome**: Table displayed, notification shown, log: `INFO - cli.interface - Query results saved to: ...`.
  - **Scenario 2**: Empty results.
    - **Example**: "Show orders from 2020-01-01" → CLI: "No data returned for query."
    - **Outcome**: No CSV, rejected query stored.
- **Validation Rules**:
  - CSV path must follow naming convention: `output_<schema>_<YYYYMMDD_HHMMSS>_<nlq_slug>.csv`.
  - Sample data must not exceed 5 rows.
  - CSV must contain all result rows without index column.
  - Log result storage (e.g., `INFO - opden.data_executor - Generated output for NLQ ...`).

## Technical Requirements

### TR1: System Architecture
**Description**: Modular architecture to support NLQ processing, data access, and error handling.
- **Components**:
  - **CLI Interface**: Handles user input/output (Python, Typer).
  - **Orchestrator**: Coordinates components (Python).
  - **NLP Processor**: Tokenizes and extracts entities (spaCy, `en_core_web_sm`).
  - **Table Identifier**: Maps tokens to tables (Azure Open AI embeddings, synonym config).
  - **Prompt Generator**: Generates SQL (Azure Open AI, `text-embedding-3-small`).
  - **Data Executor**: Executes queries (DuckDB for S3, pyodbc for SQL Server).
  - **Storage Manager**: Manages metadata and S3 access (s3fs, Python).
  - **DB Manager**: Manages database connections (pyodbc, SQLite).
- **Validation Rules**:
  - Each component must initialize without errors (e.g., log: `DEBUG - <component> - Initialized ...`).
  - Components must communicate via defined interfaces (e.g., entities dict, SQL string).
  - Log component initialization and interactions.

### TR2: S3-Compatible Object Storage
**Description**: Support data retrieval from S3-protocol-enabled object storage (e.g., Cleversafe).
- **Details**:
  - Access CSV files (e.g., `s3://bike-stores-bucket/data-files/orders.csv`).
  - Use metadata-driven schemas (e.g., `metadata_data_default.json`).
  - Load data into DuckDB in-memory tables.
  - Support authentication via access key and secret key.
- **Scenarios**:
  - **Scenario 1**: Load `orders` table.
    - **Example**: Query "Show orders from 2016-01-01" loads `orders.csv`.
    - **Outcome**: Table created with 8 columns, `order_date` cast to `DATE`.
  - **Scenario 2**: Fallback loading for misaligned CSV.
    - **Example**: Binder error → Log: `DEBUG - opden.data_executor - Fallback loaded schema for table orders: ...`.
    - **Outcome**: Table loaded with auto-detected schema.
- **Validation Rules**:
  - S3 paths must be valid (e.g., `s3://<bucket>/data-files/<table>.csv`).
  - Metadata must define columns and types.
  - Handle `Binder Error` with fallback loading.
  - Log S3 access and schema inference (e.g., `DEBUG - opden.data_executor - Inferred CSV schema: ...`).

### TR3: Azure Open AI Integration
**Description**: Use Azure Open AI for SQL generation and table identification.
- **Details**:
  - Model: `text-embedding-3-small` for embeddings.
  - Deployment: `embedding-model`.
  - API version: `2024-12-01-preview`.
  - Endpoint: Configured in `app-config/azure_config.json`.
- **Scenarios**:
  - **Scenario 1**: Generate SQL for NLQ.
    - **Example**: "Show orders from 2016-01-01" → SQL: `SELECT * FROM orders WHERE order_date = '2016-01-01';`.
    - **Outcome**: Valid SQL generated, log: `DEBUG - proga.prompt_generator - Generated SQL: ...`.
  - **Scenario 2**: Identify tables.
    - **Example**: Token `orders` → Table `default.orders`.
    - **Outcome**: Log: `DEBUG - tia.table_identifier - Table default.orders scored 0.9 for token orders`.
- **Validation Rules**:
  - API calls must succeed with valid credentials.
  - Embeddings must score tables above threshold (e.g., 0.5).
  - Log API calls and responses (e.g., `DEBUG - tia.table_identifier - Encoded 1 queries using deployment embedding-model`).

### TR4: Date Handling
**Description**: Support queries with date filters and validate date formats.
- **Details**:
  - Detect string-type date columns (e.g., `order_date`) via metadata or name (`date` in column name).
  - Cast to `DATE` in DuckDB using `TRY_CAST(strptime(column, '%Y-%m-%d') AS DATE)`.
  - Validate NLQ dates as `YYYY-MM-DD`.
  - Adjust SQL for `YEAR()` or `LIKE 'YYYY%'` to `strftime(...)`.
- **Scenarios**:
  - **Scenario 1**: Valid date query.
    - **Example**: "Show orders from 2016-01-01" → SQL adjusted for `order_date`.
    - **Outcome**: Results returned, log: `DEBUG - opden.data_executor - Adjusted SQL query for date columns: ...`.
  - **Scenario 2**: Invalid date format.
    - **Example**: "Show orders from 2016" → Error: "Invalid date format '2016'. Please use YYYY-MM-DD (e.g., 2016-01-01)."
    - **Outcome**: Rejected query stored, log: `ERROR - opden.data_executor - Invalid date format '2016'. ...`.
  - **Scenario 3**: Invalid date value.
    - **Example**: "Show orders from 2016-13-01" → Same error as above.
    - **Outcome**: Rejected query stored.
- **Validation Rules**:
  - Dates must match `YYYY-MM-DD` via regex and `pd.to_datetime`.
  - Date columns must be identified in metadata or by name.
  - Log date validation and SQL adjustments.

### TR5: Logging and Monitoring
**Description**: Log system activities and errors for debugging and auditing.
- **Details**:
  - Log file: `logs/datascriber.log`.
  - Levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`.
  - Store rejected queries in SQLite: `data/<datasource>/datascriber.db`.
- **Scenarios**:
  - **Scenario 1**: Successful query.
    - **Example**: "Show orders from 2016-01-01" → Log: `INFO - cli.interface - Query processed successfully: ...`.
    - **Outcome**: Full execution trace logged.
  - **Scenario 2**: Error logging.
    - **Example**: "Show orders from 2016" → Log: `ERROR - opden.data_executor - Invalid date format '2016'. ...`.
    - **Outcome**: Stack trace and rejected query stored.
- **Validation Rules**:
  - Logs must include timestamp, component, level, and message.
  - Rejected queries must store all required fields.
  - Log file must be created at startup (e.g., `DEBUG - root - File logging initialized to: ...`).

### TR6: Configuration Management
**Description**: Manage configurations for datasources, LLM, and cloud services.
- **Details**:
  - Files: `app-config/db_configurations.json`, `llm_config.json`, `azure_config.json`, `aws_config.json`, `synonym_config.json`.
  - Directories: `app-config/`, `data/`, `models/`, `logs/`, `temp/`.
- **Scenarios**:
  - **Scenario 1**: Load configurations.
    - **Example**: Startup → Log: `DEBUG - main - Validated configuration file: db_configurations.json`.
    - **Outcome**: All configs loaded, directories created.
  - **Scenario 2**: Missing config.
    - **Example**: Missing `llm_config.json` → Error: "Failed to load llm_config.json."
    - **Outcome**: System halts, log error.
- **Validation Rules**:
  - Config files must be valid JSON.
  - Required fields: datasource name, type, schemas; LLM model; Azure endpoint; AWS credentials.
  - Log config validation (e.g., `DEBUG - main - All configuration files validated successfully`).

### TR7: Dependency Management
**Description**: Ensure compatibility with required libraries.
- **Details**:
  - Python: 3.11.10.
  - Key libraries: `duckdb==1.3.0`, `pandas==2.0.3`, `s3fs==2023.12.2`, `spacy==3.8.0`, `azure-ai-textanalytics==5.3.0`, `pyodbc==5.0.1`.
- **Validation Rules**:
  - Log dependencies at startup (e.g., `DEBUG - main - Python: 3.11.10, Dependencies: ...`).
  - Fail startup if critical dependencies are missing.

## Non-Functional Requirements

### NFR1: Performance
- **Requirement**: Process NLQs within 10 seconds for datasets with up to 10,000 rows.
- **Validation**: Log query execution time (e.g., `DEBUG - opden.data_executor - Query execution completed ...`).

### NFR2: Scalability
- **Requirement**: Handle up to 100 concurrent users with minimal latency.
- **Validation**: Test with load simulation, monitor logs for bottlenecks.

### NFR3: Security
- **Requirement**: Securely store and access credentials (e.g., AWS access keys, Azure API keys).
- **Validation**: Credentials must be read from config files, not hardcoded. Log access attempts.

### NFR4: Reliability
- **Requirement**: Achieve 99.9% uptime with robust error recovery.
- **Validation**: Test fallback loading and error handling. Monitor `rejected_queries` for patterns.

## Assumptions
- S3-compatible storage supports standard S3 protocol.
- Azure Open AI services are accessible via token-based authentication.
- Organization-specific tools will handle deployment, authentication customization, and Client ID setup.
- CSV files follow consistent structure with headers.

## References
- Log file: `logs/datascriber.log` (e.g., entries from 2025-06-17).
- Example queries: "Show orders from 2016-01-01", "Show orders from 2016", "Show customers in NY".
- Metadata: `data/bikestores_s3/metadata_data_default.json` (11 tables, e.g., `orders` with 8 columns).