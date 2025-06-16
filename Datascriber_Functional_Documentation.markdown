# Datascriber Functional Documentation

## Overview
Datascriber is a natural language query (NLQ) processing system that converts user queries into SQL for execution against SQL Server or S3 datasources. It supports schema validation, metadata-driven table loading, and dynamic date handling for S3-based CSV files.

## System Architecture
- **CLI Interface**: Handles user input, displays results, and manages query mode.
- **Orchestrator**: Coordinates NLQ processing, table identification, and query execution.
- **NLP Processor**: Tokenizes queries and extracts entities (e.g., dates, objects).
- **Table Identifier**: Maps NLQ tokens to tables using synonyms and embeddings.
- **Prompt Generator**: Generates SQL queries using Azure OpenAI.
- **Data Executor**: Executes SQL queries, handles S3 data loading with DuckDB, and adjusts queries for date columns.
- **Storage Manager**: Manages metadata, S3 access, and rejected queries.
- **DB Manager**: Manages SQL Server connections and local SQLite storage.

## Key Features
1. **NLQ to SQL Conversion**:
   - Processes queries like "Show orders from 2016-01-01" to generate SQL: `SELECT * FROM orders WHERE order_date = '2016-01-01';`.
   - Uses metadata for table and column mapping.
2. **S3 Data Loading**:
   - Loads CSV files from S3 (e.g., `s3://bike-stores-bucket/data-files/orders.csv`) into DuckDB in-memory tables.
   - Enforces metadata-driven schemas with explicit column names.
   - Auto-detects CSV delimiters and headers.
   - Supports fallback loading if metadata schema fails.
3. **Date Handling**:
   - Detects string-type date columns (e.g., `order_date`) in metadata.
   - Casts dates to `DATE` during table loading using `TRY_CAST(strptime(column, '%Y-%m-%d') AS DATE)`.
   - Adjusts SQL queries for date comparisons (e.g., `YEAR(order_date)` to `strftime(...)`).
   - Validates date formats in NLQ (requires `YYYY-MM-DD`).
4. **Error Handling**:
   - Logs errors to `logs/datascriber.log`.
   - Stores rejected queries in `data/<datasource>/datascriber.db` with error types (e.g., `INVALID_DATE_FORMAT`).
   - Provides user-friendly messages for invalid inputs.

## Usage Guidelines
### Query Syntax
- Use natural language queries, e.g., "Show orders from 2016-01-01", "List customers in NY".
- Specify dates in `YYYY-MM-DD` format for tables with date columns (e.g., `orders.order_date`).
- Avoid ambiguous terms unless defined in `synonym_config.json`.

### Date Query Hints
- **Valid Examples**:
  - "Show orders from 2016-01-01" → Matches `order_date = '2016-01-01'`.
  - "Show orders between 2016-01-01 and 2016-01-03" → Uses `order_date BETWEEN '2016-01-01' AND '2016-01-03'`.
- **Invalid Examples**:
  - "Show orders from 2016" → Fails with "Invalid date format '2016'. Please use YYYY-MM-DD (e.g., 2016-01-01)."
  - "Show orders from 2016-13-01" → Fails with "Invalid date format '2016-13-01'. Please use YYYY-MM-DD (e.g., 2016-01-01)."
- **Tip**: Always use `YYYY-MM-DD` for date filters to ensure compatibility with string-type date columns.

## Configuration
- **db_configurations.json**: Defines datasources (e.g., `bikestores_s3` with schema `default`).
- **llm_config.json**: Configures Azure OpenAI and date format validation (expects `YYYY-MM-DD`).
- **aws_config.json**: Provides S3 credentials for `bikestores_s3`.
- **metadata_data_default.json**: Defines table schemas (e.g., `orders` with 8 columns).

## Error Handling
- **Invalid Date Format**: Rejects queries with non-`YYYY-MM-DD` dates and displays a user-friendly message.
- **Binder Error**: Handled by metadata-driven schema enforcement and fallback loading.
- **No Data**: Logs warning and stores in `rejected_queries` with `NO_DATA`.
- **DuckDB Errors**: Logged with stack traces and stored as `DUCKDB_ERROR`.

## Logging
- Location: `logs/datascriber.log`
- Includes:
  - Inferred and loaded table schemas.
  - Adjusted SQL queries.
  - Query execution results and errors.
  - Component initialization and validation.

## Dependencies
- Python 3.11.10
- Key libraries: `duckdb==1.3.0`, `pandas==2.0.3`, `s3fs==2023.12.2`, `spacy==3.8.0`, `azure-ai-textanalytics==5.3.0`