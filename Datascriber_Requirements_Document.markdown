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
- Advanced NLP, table identification, and prompt generation using Azure Open AI embeddings.

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
  - **Scenario 1**: Admin logs in.
    - **Example**: `login admin` → "User admin logged in."
    - **Outcome**: Session initialized, user can select datasource.
  - **Scenario 2**: Invalid username.
    - **Example**: `login invalid_user` → "Invalid username."
    - **Outcome**: Access denied, prompt for retry.
- **Validation Rules**:
  - Username must be a non-empty string.
  - Username must exist in configuration (e.g., `app-config/users.json`).
  - Log successful logins (e.g., `INFO - core.orchestrator - User admin logged in`).
  - Log failed attempts (e.g., `WARNING - core.orchestrator - Invalid username: invalid_user`).

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
  - **Scenario 2**: Invalid datasource.
    - **Example**: `select-datasource invalid_ds` → "Invalid datasource: invalid_ds."
    - **Outcome**: Prompt for retry, log error.
- **Validation Rules**:
  - Datasource must exist in `db_configurations.json`.
  - Schemas must be non-empty and valid.
  - Metadata file must exist and contain valid JSON.
  - Log schema and metadata loading (e.g., `DEBUG - config.utils - Datasource bikestores_s3: Loaded schemas ['default']`).

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

### FR4: NLQ Processing
**Description**: Process NLQs to generate and execute SQL queries.
- **Steps**:
  1. User submits NLQ (e.g., "Show orders from 2016-01-01").
  2. NLP Processor tokenizes and extracts entities.
  3. Table Identifier maps tokens to tables using synonyms and embeddings.
  4. Prompt Generator creates SQL query using Azure Open AI.
  5. Data Executor validates and adjusts SQL (e.g., for date columns).
  6. Execute query against datasource.
  7. Save results to CSV (e.g., `temp/query_results/output_default_*.csv`).
  8. Display sample data (first 5 rows).
- **Scenarios**:
  - **Scenario 1**: Valid date query.
    - **Example**: "Show orders from 2016-01-01" → SQL: `SELECT * FROM orders WHERE order_date = '2016-01-01';`.
    - **Outcome**: CSV with 2 rows, sample data displayed.
  - **Scenario 2**: Invalid date format.
    - **Example**: "Show orders from 2016" → Error: "Invalid date format '2016'. Please use YYYY-MM-DD."
    - **Outcome**: No CSV, rejected query stored.
  - **Scenario 3**: Non-date query.
    - **Example**: "Show customers in NY" → SQL: `SELECT * FROM customers WHERE state = 'NY';`.
    - **Outcome**: CSV with matching rows.
- **Validation Rules**:
  - NLQ must be non-empty.
  - Dates must match `YYYY-MM-DD`.
  - SQL query must be valid `SELECT`.
  - Log entity extraction, SQL generation, and execution.

### FR5: Error Handling and Feedback
**Description**: Provide user-friendly error messages and log detailed errors.
- **Steps**:
  1. Detect error (e.g., invalid date, missing table).
  2. Display CLI message (e.g., "Invalid date format '2016'. ...").
  3. Log error with stack trace to `logs/datascriber.log`.
  4. Store rejected query in `data/<datasource>/datascriber.db`.
- **Scenarios**:
  - **Scenario 1**: Invalid date format.
    - **Example**: "Show orders from 2016" → CLI: "Invalid date format '2016'. ...".
    - **Outcome**: Rejected query stored with `error_type="INVALID_DATE_FORMAT"`.
  - **Scenario 2**: Binder error.
    - **Example**: Misaligned CSV → Log: `ERROR - opden.data_executor - DuckDB error: Binder Error: ...`.
    - **Outcome**: Fallback loading attempted, rejected query stored.
- **Validation Rules**:
  - CLI messages must be actionable.
  - Logs must include error context.
  - Rejected queries must include all required fields (e.g., `error_type`).

### FR6: Result Storage and Display
**Description**: Save query results to CSV and display sample data.
- **Steps**:
  1. Retrieve results as DataFrame.
  2. Save to `temp/query_results/output_<schema>_<timestamp>_<nlq_slug>.csv`.
  3. Display first 5 rows in CLI.
  4. Notify user: "Query results saved to: <csv_path>."
- **Scenarios**:
  - **Scenario 1**: Valid query.
    - **Example**: "Show orders from 2016-01-01" → CSV: `output_default_20250617_020608_show_orders_from_2016_01_01.csv`.
    - **Outcome**: Table displayed, notification shown.
  - **Scenario 2**: Empty results.
    - **Example**: "Show orders from 2020-01-01" → CLI: "No data returned."
    - **Outcome**: No CSV, rejected query stored.
- **Validation Rules**:
  - CSV must follow naming convention.
  - Sample data limited to 5 rows.
  - Log result storage.

## Technical Requirements

### TR1: System Architecture
**Description**: Modular architecture for NLQ processing and data access.
- **Components**:
  - CLI Interface (Typer).
  - Orchestrator (Python).
  - NLP Processor (spaCy, `en_core_web_sm`).
  - Table Identifier (Azure Open AI embeddings).
  - Prompt Generator (Azure Open AI).
  - Data Executor (DuckDB, pyodbc).
  - Storage Manager (s3fs).
  - DB Manager (pyodbc, SQLite).
- **Validation Rules**:
  - Components must initialize without errors.
  - Log interactions (e.g., `DEBUG - <component> - Initialized ...`).

### TR2: S3-Compatible Object Storage
**Description**: Support data retrieval from S3-protocol-enabled storage.
- **Details**:
  - Access CSV files (e.g., `s3://bike-stores-bucket/data-files/orders.csv`).
  - Load into DuckDB with metadata-driven schemas.
  - Support authentication via access key/secret key.
- **Scenarios**:
  - **Scenario 1**: Load `orders` table.
    - **Example**: "Show orders from 2016-01-01" → Loads `orders.csv`.
    - **Outcome**: Table created with 8 columns.
  - **Scenario 2**: Fallback loading.
    - **Example**: Binder error → Log: `DEBUG - opden.data_executor - Fallback loaded schema ...`.
    - **Outcome**: Auto-detected schema used.
- **Validation Rules**:
  - S3 paths must be valid.
  - Metadata must define columns/types.
  - Log S3 access and schema inference.

### TR3: Azure Open AI Integration
**Description**: Use Azure Open AI for SQL generation and table identification.
- **Details**:
  - Model: `text-embedding-3-small`.
  - Deployment: `embedding-model`.
  - API version: `2024-12-01-preview`.
- **Scenarios**:
  - **Scenario 1**: Generate SQL.
    - **Example**: "Show orders from 2016-01-01" → SQL: `SELECT * FROM orders WHERE order_date = '2016-01-01';`.
    - **Outcome**: Valid SQL generated.
  - **Scenario 2**: Identify tables.
    - **Example**: Token `orders` → Table `default.orders`.
    - **Outcome**: Log: `DEBUG - tia.table_identifier - Table default.orders scored 0.9 ...`.
- **Validation Rules**:
  - API calls must succeed.
  - Log API responses.

### TR4: Date Handling
**Description**: Support date filters and validate formats.
- **Details**:
  - Detect string-type date columns (e.g., `order_date`).
  - Cast to `DATE` in DuckDB.
  - Validate NLQ dates as `YYYY-MM-DD`.
- **Scenarios**:
  - **Scenario 1**: Valid date.
    - **Example**: "Show orders from 2016-01-01" → Adjusted SQL.
    - **Outcome**: Results returned.
  - **Scenario 2**: Invalid date.
    - **Example**: "Show orders from 2016" → Error: "Invalid date format '2016'. ...".
    - **Outcome**: Rejected query stored.
- **Validation Rules**:
  - Dates must match `YYYY-MM-DD`.
  - Log date validation.

### TR5: Logging and Monitoring
**Description**: Log activities and errors.
- **Details**:
  - Log file: `logs/datascriber.log`.
  - Store rejected queries in SQLite.
- **Scenarios**:
  - **Scenario 1**: Successful query.
    - **Example**: "Show orders from 2016-01-01" → Log: `INFO - cli.interface - Query processed successfully`.
    - **Outcome**: Execution trace logged.
  - **Scenario 2**: Error.
    - **Example**: "Show orders from 2016" → Log: `ERROR - opden.data_executor - Invalid date format ...`.
    - **Outcome**: Rejected query stored.
- **Validation Rules**:
  - Logs must include timestamp, component, level.
  - Rejected queries must store required fields.

### TR6: Configuration Management
**Description**: Manage configurations.
- **Details**:
  - Files: `db_configurations.json`, `llm_config.json`, `azure_config.json`, `aws_config.json`, `synonym_config.json`.
  - Directories: `app-config/`, `data/`, `models/`, `logs/`, `temp/`.
- **Scenarios**:
  - **Scenario 1**: Load configs.
    - **Example**: Startup → Log: `DEBUG - main - Validated configuration file: ...`.
    - **Outcome**: Configs loaded.
  - **Scenario 2**: Missing config.
    - **Example**: Missing `llm_config.json` → Error: "Failed to load ...".
    - **Outcome**: System halts.
- **Validation Rules**:
  - Configs must be valid JSON.
  - Log validation.

### TR7: Dependency Management
**Description**: Ensure library compatibility.
- **Details**:
  - Python: 3.11.10.
  - Libraries: `duckdb==1.3.0`, `pandas==2.0.3`, `s3fs==2023.12.2`, `spacy==3.8.0`.
- **Validation Rules**:
  - Log dependencies at startup.
  - Fail if critical dependencies are missing.

### TR8: Table Identification Logic
**Description**: Map NLQ tokens to database tables using synonyms and embeddings.
- **Details**:
  - Use spaCy (`en_core_web_sm`) to tokenize NLQ.
  - Map tokens to tables via `synonym_config.json` (e.g., `orders: ['order', 'purchases']`).
  - Compute cosine similarity between token embeddings and table/column embeddings (Azure Open AI, `text-embedding-3-small`).
  - Select tables with similarity above threshold (e.g., 0.5).
  - Fallback to metadata table names if no match.
- **Scenarios**:
  - **Scenario 1**: Direct synonym match.
    - **Example**: "Show orders from 2016-01-01".
    - **Technical Example**:
      - Tokens: `['orders', '2016-01-01']`.
      - Synonym mapping: `orders` → `orders` (`synonym_config.json`).
      - Output: `default.orders`.
      - Log: `DEBUG - nlp.nlp_processor - Mapped token 'orders' to 'orders'`.
    - **Outcome**: Table `orders` identified.
  - **Scenario 2**: Embedding-based match.
    - **Example**: "Show purchases in 2016".
    - **Technical Example**:
      - Tokens: `['purchases', '2016']`.
      - No direct synonym match.
      - Embeddings: `purchases` → `[0.1, 0.2, ...]`, `orders` → `[0.12, 0.19, ...]`.
      - Cosine similarity: `cos_sim(purchases, orders) = 0.92` (above 0.5).
      - Output: `default.orders`.
      - Log: `DEBUG - tia.table_identifier - Table default.orders scored 0.92 for token purchases`.
    - **Outcome**: Table `orders` identified.
  - **Scenario 3**: No match.
    - **Example**: "Show sales".
    - **Technical Example**:
      - Tokens: `['sales']`.
      - No synonym or embedding match (highest similarity < 0.5).
      - Output: None.
      - Log: `WARNING - tia.table_identifier - No tables identified for token 'sales'`.
    - **Outcome**: Error: "No tables identified for query."
- **Validation Rules**:
  - Tokens must be non-empty.
  - Synonym mappings must be valid JSON.
  - Embeddings must be 1536-dimensional (Azure Open AI default).
  - Log token mapping and similarity scores.
  - Store rejected queries with `error_type="NO_TABLES"`.

### TR9: Prompt Generation Logic
**Description**: Generate SQL queries from NLQs using Azure Open AI.
- **Details**:
  - Build context: NLQ, entities, predicted tables, metadata DDL.
  - Use system prompt with schema details (e.g., `CREATE TABLE orders (...);`).
  - Call Azure Open AI (`text-embedding-3-small`) to generate SQL.
  - Validate SQL syntax (starts with `SELECT`, ends with `;`).
- **Scenarios**:
  - **Scenario 1**: Simple query.
    - **Example**: "Show orders from 2016-01-01".
    - **Technical Example**:
      - Context: `NLQ: Show orders from 2016-01-01; Entities: {'dates': ['2016-01-01'], 'objects': ['orders']}; Tables: default.orders; DDL: CREATE TABLE orders (order_id INTEGER, order_date STRING, ...);`.
      - Prompt: `Given schema default with table orders (order_id INTEGER, order_date STRING, ...), generate SQL for: Show orders from 2016-01-01`.
      - Output: `SELECT * FROM orders WHERE order_date = '2016-01-01';`.
      - Log: `DEBUG - proga.prompt_generator - Generated SQL: SELECT * FROM orders ...`.
    - **Outcome**: Valid SQL generated.
  - **Scenario 2**: Complex query.
    - **Example**: "Show customers in NY with orders in 2016".
    - **Technical Example**:
      - Context: `Entities: {'places': ['NY'], 'objects': ['customers', 'orders'], 'dates': ['2016']}; Tables: default.customers, default.orders`.
      - Prompt: `Generate SQL for: Show customers in NY with orders in 2016`.
      - Output: `SELECT c.* FROM customers c JOIN orders o ON c.customer_id = o.customer_id WHERE c.state = 'NY' AND YEAR(o.order_date) = '2016';`.
      - Log: `DEBUG - proga.prompt_generator - Generated SQL: SELECT c.* FROM customers ...`.
    - **Outcome**: SQL with JOIN generated.
- **Validation Rules**:
  - Prompt must include NLQ, entities, and DDL.
  - SQL must be valid `SELECT`.
  - Log prompt and SQL output.
  - Store rejected queries with `error_type="NO_SQL_GENERATED"`.

### TR10: S3 Files Processing Logic
**Description**: Load S3 CSV files into DuckDB in-memory tables.
- **Details**:
  - Use s3fs to access files (e.g., `s3://bike-stores-bucket/data-files/orders.csv`).
  - Load with metadata-driven schema (e.g., `order_id INTEGER, order_date STRING`).
  - Cast date columns to `DATE` using `TRY_CAST(strptime(column, '%Y-%m-%d') AS DATE)`.
  - Fallback to auto-detected schema on Binder Error.
- **Scenarios**:
  - **Scenario 1**: Standard loading.
    - **Example**: "Show orders from 2016-01-01".
    - **Technical Example**:
      - S3 path: `s3://bike-stores-bucket/data-files/orders.csv`.
      - Metadata: `columns: [{'name': 'order_id', 'type': 'INTEGER'}, {'name': 'order_date', 'type': 'STRING'}]`.
      - SQL: `CREATE TABLE orders AS SELECT order_id, TRY_CAST(strptime(order_date, '%Y-%m-%d') AS DATE) AS order_date, ... FROM read_csv('s3://...', header=true, columns=['order_id', 'order_date', ...]);`.
      - Log: `INFO - opden.data_executor - Successfully loaded table orders ...`.
    - **Outcome**: Table loaded with 8 columns.
  - **Scenario 2**: Fallback loading.
    - **Example**: Binder Error due to misaligned columns.
    - **Technical Example**:
      - SQL: `CREATE TABLE orders AS SELECT * FROM read_csv('s3://...', header=true, auto_detect=true);`.
      - Log: `DEBUG - opden.data_executor - Fallback loaded schema for table orders: ...`.
    - **Outcome**: Auto-detected schema used.
- **Validation Rules**:
  - S3 paths must be accessible.
  - Metadata must match CSV structure.
  - Log schema inference and loading errors.
  - Store rejected queries with `error_type="DATA_LOAD_ERROR"`.

### TR11: Metadata Creation Logic
**Description**: Generate metadata for tables and columns.
- **Details**:
  - Extract schema from CSV files or SQL Server tables.
  - Store in `data/<datasource>/metadata_data_<schema>.json`.
  - Include: table name, description, columns (name, type, unique_values, synonyms).
- **Scenarios**:
  - **Scenario 1**: CSV metadata creation.
    - **Example**: `orders.csv`.
    - **Technical Example**:
      - Input: `orders.csv` with columns `order_id, order_date, ...`.
      - Process: Read first 100 rows, infer types (e.g., `order_id: INTEGER`, `order_date: STRING`).
      - Output: `metadata_data_default.json`:
        ```json
        {
          "tables": [
            {
              "name": "orders",
              "description": "",
              "columns": [
                {"name": "order_id", "type": "INTEGER", "unique_values": [1, 2, ...], "synonyms": []},
                {"name": "order_date", "type": "STRING", "unique_values": ["2016-01-01", ...], "synonyms": []}
              ]
            }
          ]
        }
        ```
      - Log: `DEBUG - config.utils - Created metadata for schema default`.
    - **Outcome**: Metadata file created.
- **Validation Rules**:
  - Metadata must be valid JSON.
  - Column types must match data (e.g., `INTEGER`, `STRING`).
  - Log metadata creation.

### TR12: Metadata Usage Logic
**Description**: Use metadata to drive table loading and query execution.
- **Details**:
  - Load metadata from `data/<datasource>/metadata_data_<schema>.json`.
  - Use for schema validation, table identification, and SQL adjustment.
- **Scenarios**:
  - **Scenario 1**: Table loading.
    - **Example**: "Show orders from 2016-01-01".
    - **Technical Example**:
      - Metadata: `columns: [{'name': 'order_date', 'type': 'STRING', 'unique_values': ['2016-01-01']}]`.
      - Action: Cast `order_date` to `DATE` in DuckDB.
      - Log: `DEBUG - opden.data_executor - Detected date-like column orders.order_date`.
    - **Outcome**: Table loaded correctly.
  - **Scenario 2**: Query adjustment.
    - **Example**: "Show orders in 2016".
    - **Technical Example**:
      - Metadata: Identifies `order_date` as date column.
      - Action: Adjust SQL: `YEAR(order_date)` → `strftime(TRY_CAST(strptime(order_date, '%Y-%m-%d') AS DATE), '%Y')`.
      - Log: `DEBUG - opden.data_executor - Adjusted SQL query for date columns ...`.
    - **Outcome**: SQL adjusted.
- **Validation Rules**:
  - Metadata must be loaded before query execution.
  - Log metadata access.

### TR13: Rich Metadata Processing Logic
**Description**: Process rich metadata (e.g., unique_values, synonyms) for enhanced query accuracy.
- **Details**:
  - Use `unique_values` to detect date columns (e.g., `2016-01-01`).
  - Use `synonyms` for table/column mapping.
  - Validate metadata integrity (non-empty tables, valid types).
- **Scenarios**:
  - **Scenario 1**: Date column detection.
    - **Example**: `order_date` with `unique_values: ['2016-01-01', '2016-01-02']`.
    - **Technical Example**:
      - Process: Regex `^\d{4}-\d{2}-\d{2}$` matches values.
      - Action: Flag `order_date` as date column.
      - Log: `DEBUG - opden.data_executor - Detected date-like column orders.order_date`.
    - **Outcome**: Date handling applied.
  - **Scenario 2**: Synonym mapping.
    - **Example**: `customers` with `synonyms: ['customer', 'clients']`.
    - **Technical Example**:
      - NLQ: "Show clients in NY".
      - Action: Map `clients` to `customers`.
      - Log: `DEBUG - nlp.nlp_processor - Mapped token 'clients' to 'customers'`.
    - **Outcome**: Table `customers` identified.
- **Validation Rules**:
  - `unique_values` must be representative (e.g., top 100 values).
  - Synonyms must be non-empty lists.
  - Log metadata processing.

### TR14: Synonyms Usage Logic
**Description**: Map NLQ tokens to tables/columns using synonyms.
- **Details**:
  - Load from `app-config/synonym_config.json` (e.g., `orders: ['order', 'purchases']`).
  - Match tokens case-insensitively.
  - Fallback to embeddings if no synonym match.
- **Scenarios**:
  - **Scenario 1**: Synonym match.
    - **Example**: "Show purchases from 2016-01-01".
    - **Technical Example**:
      - Token: `purchases`.
      - Synonym: `orders: ['order', 'purchases']`.
      - Output: `default.orders`.
      - Log: `DEBUG - nlp.nlp_processor - Mapped token 'purchases' to 'orders'`.
    - **Outcome**: Table identified.
  - **Scenario 2**: No synonym match.
    - **Example**: "Show sales".
    - **Technical Example**:
      - Token: `sales`.
      - No synonym match.
      - Action: Use embeddings.
      - Log: `DEBUG - nlp.nlp_processor - No synonym mapping found for token 'sales'`.
    - **Outcome**: Fallback to embeddings.
- **Validation Rules**:
  - Synonyms must be valid JSON.
  - Log synonym mappings.

### TR15: Usage of Cosine Similarity
**Description**: Compute similarity between NLQ tokens and table/column names.
- **Details**:
  - Generate embeddings using Azure Open AI (`text-embedding-3-small`).
  - Compute cosine similarity: `cos_sim(a, b) = (a·b) / (||a||·||b||)`.
  - Select matches above threshold (e.g., 0.5).
- **Scenarios**:
  - **Scenario 1**: High similarity.
    - **Example**: "Show purchases".
    - **Technical Example**:
      - Token embedding: `purchases` → `[0.1, 0.2, ...]`.
      - Table embedding: `orders` → `[0.12, 0.19, ...]`.
      - Cosine similarity: `0.92`.
      - Output: `default.orders`.
      - Log: `DEBUG - tia.table_identifier - Table default.orders scored 0.92 ...`.
    - **Outcome**: Table identified.
  - **Scenario 2**: Low similarity.
    - **Example**: "Show sales".
    - **Technical Example**:
      - Token embedding: `sales` → `[0.3, 0.4, ...]`.
      - Table embeddings: All scores < 0.5.
      - Output: None.
      - Log: `WARNING - tia.table_identifier - No tables scored above threshold`.
    - **Outcome**: Error: "No tables identified."
- **Validation Rules**:
  - Embeddings must be normalized.
  - Threshold must be configurable (e.g., `model_config.json`).
  - Log similarity scores.

### TR16: Usage of NLP
**Description**: Process NLQs using NLP for tokenization and entity extraction.
- **Details**:
  - Use spaCy (`en_core_web_sm`) for tokenization and NER.
  - Extract entities: dates, objects, places, names.
  - Map tokens to synonyms or embeddings.
- **Scenarios**:
  - **Scenario 1**: Entity extraction.
    - **Example**: "Show orders from 2016-01-01 in NY".
    - **Technical Example**:
      - spaCy output: Tokens: `['orders', 'from', '2016-01-01', 'in', 'NY']`.
      - Entities: `{'dates': ['2016-01-01'], 'objects': ['orders'], 'places': ['NY']}`.
      - Log: `DEBUG - nlp.nlp_processor - Extracted entities: {...}`.
    - **Outcome**: Entities used for table identification and SQL generation.
  - **Scenario 2**: Invalid date.
    - **Example**: "Show orders from 2016".
    - **Technical Example**:
      - Entities: `{'dates': ['2016'], 'objects': ['orders']}`.
      - Validation: `2016` fails `YYYY-MM-DD` check.
      - Log: `ERROR - opden.data_executor - Invalid date format '2016'. ...`.
    - **Outcome**: Error displayed.
- **Validation Rules**:
  - Tokens must be non-empty.
  - Entities must be categorized correctly.
  - Log tokenization and entity extraction.

### TR17: Usage of Sentence Embedders
**Description**: Generate embeddings for NLQ tokens and table/column names.
- **Details**:
  - Use Azure Open AI (`text-embedding-3-small`, 1536 dimensions).
  - Embed tokens and table/column names.
  - Store embeddings in `models/model_<datasource>.json` (if caching enabled).
- **Scenarios**:
  - **Scenario 1**: Token embedding.
    - **Example**: "Show orders".
    - **Technical Example**:
      - Token: `orders`.
      - API call: `embeddings.create(model='embedding-model', input='orders')`.
      - Output: `[0.1, 0.2, ..., 0.1536]`.
      - Log: `DEBUG - tia.table_identifier - Encoded 1 queries using deployment embedding-model`.
    - **Outcome**: Embedding used for cosine similarity.
  - **Scenario 2**: Empty embeddings cache.
    - **Example**: "Show purchases".
    - **Technical Example**:
      - Cache: `model_bikestores_s3.json` empty.
      - Action: Generate embeddings for all tables.
      - Log: `WARNING - tia.table_identifier - Model at ... has empty embeddings`.
    - **Outcome**: Embeddings generated and cached.
- **Validation Rules**:
  - Embeddings must be 1536-dimensional.
  - API calls must succeed.
  - Log embedding generation.

## Non-Functional Requirements

### NFR1: Performance
- **Requirement**: Process NLQs within 10 seconds for datasets up to 10,000 rows.
- **Validation**: Log execution time.

### NFR2: Scalability
- **Requirement**: Handle 100 concurrent users.
- **Validation**: Test with load simulation.

### NFR3: Security
- **Requirement**: Securely store credentials.
- **Validation**: Credentials in config files, log access attempts.

### NFR4: Reliability
- **Requirement**: 99.9% uptime.
- **Validation**: Test error recovery, monitor `rejected_queries`.

## Assumptions
- S3-compatible storage supports standard S3 protocol.
- Azure Open AI services are accessible via token-based authentication.
- Organization-specific tools handle deployment and authentication customization.
- CSV files have consistent structure with headers.

## References
- Log file: `logs/datascriber.log` (2025-06-17).
- Example queries: "Show orders from 2016-01-01", "Show orders from 2016", "Show customers in NY".
- Metadata: `data/bikestores_s3/metadata_data_default.json`.