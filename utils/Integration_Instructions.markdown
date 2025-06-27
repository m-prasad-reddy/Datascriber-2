# Datascriber 2.0 Integration Instructions

This document provides instructions for setting up, configuring, and executing Datascriber 2.0, an enhanced Text-to-SQL system that converts natural language queries (NLQs) into SQL queries for SQL Server (`bikestores`) and S3 (`salesdata`) datasources. New features include DuckDB for S3 query execution and improved date handling for string-type date columns (e.g., `order_date`). It supports Data User and Admin User roles, leveraging Azure Open AI (`gpt-4o`, `text-embedding-3-small`), bulk training, and TIA 1.2.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Configuration](#configuration)
- [File Placement](#file-placement)
- [Execution](#execution)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Optional Enhancements](#optional-enhancements)

## Prerequisites

- **Environment**:
  - Python 3.8 or higher.
  - OS: Windows, Linux, or macOS.
  - Git for repository cloning.
  - SQLite client for Admin User tasks.
- **Accounts**:
  - Azure Open AI account with `gpt-4o` and `text-embedding-3-small`.
  - AWS account with access to `bike-stores-bucket`.
- **Datasources**:
  - **SQL Server (`bikestores`)**: Instance with `sales` and `production` schemas, tables: `orders`, `order_items`, `customers`, `stores`, `staffs`, `products`, `categories`, `brands`, `stocks`.
  - **S3 (`salesdata`)**: Bucket `bike-stores-bucket` with CSV files in `data-files/` (e.g., `orders.csv`, `products.csv`).

## Setup

1. **Clone Repository**:
   ```bash
   git clone https://github.com/m-prasad-reddy/Datascriber-2.git
   cd Datascriber-2
   ```

2. **Create Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Create `requirements.txt`**:
   Save to `requirements.txt`:
   ```
   spacy==3.8.0
   openai==1.86.0
   numpy==1.24.3
   pandas==2.0.3
   pyodbc==5.0.1
   boto3==1.34.0
   pyarrow==17.0.0
   python-dotenv==1.0.0
   duckdb==1.1.2
   s3fs==2023.12.2
   ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Download SpaCy Model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

6. **Create Directories**:
   ```bash
   mkdir -p app-config data logs cli core tia proga opden nlp storage docs
   ```

## Configuration

Place configuration files in `app-config/`, replacing placeholders with credentials.

1. **`db_configurations.json`**:
   ```json
   {
     "datasources": [
       {
         "name": "bikestores",
         "type": "sqlserver",
         "connection": {
           "host": "localhost",
           "database": "bikestores",
           "username": "sa",
           "password": "<your-secure-password>",
           "schemas": ["sales", "production"],
           "system_schemas": ["sys", "information_schema"]
         }
       },
       {
         "name": "salesdata",
         "type": "s3",
         "connection": {
           "bucket_name": "bike-stores-bucket",
           "database": "data-files",
           "region": "us-east-1",
           "schemas": ["default"],
           "orc_pattern": "^data_"
         }
       }
     ]
   }
   ```

2. **`llm_config.json`**:
   ```json
   {
     "model_name": "gpt-4o",
     "embedding_model": "text-embedding-3-small",
     "api_version": "2023-10-01-preview",
     "prompt_settings": {
       "system_prompt": "You are a SQL expert generating valid SQL queries for SQL Server and S3 datasources. Use CAST for dates, LOWER() and LIKE for strings, SUM() and AVG() for numerics. Ensure queries are optimized and safe.",
       "max_prompt_length": 4000,
       "max_tokens": 1000,
       "temperature": 0.1,
       "validation": {
         "enabled": true,
         "date_formats": [
           {"pattern": "\\d{4}-\\d{2}-\\d{2}", "strftime": "%Y-%m-%d"},
           {"pattern": "\\d{2}/\\d{2}/\\d{4}", "strftime": "%m/%d/%Y"},
           {"pattern": "\\d{2}-\\d{2}-\\d{4}", "strftime": "%d-%Y"}
         ],
         "error_message": "Invalid date format. Use YYYY-MM-DD, MM/DD/YYYY, or DD-MM-DDYYYY.",
         "entities": ["dates", "objects", "places"]
       }
     },
     "training_settings": {
       "enabled": true,
       "max_rows": 100,
       "fields": [
         "db_source_type",
         "db_name",
         "user_query",
         "related_tables",
         "specific_columns",
         "extracted_values",
         "placeholders",
         "relevant_sql",
         "is_slm_trained",
         "scenario_id"
       ]
     },
     "mock_enabled": true
   }
   ```

3. **`model_config.json`**:
   ```json
   {
     "sentence_transformer": {
       "model_name": "text-embedding-3-small",
       "deployment": "embedding-model",
       "enabled": true
     },
     "bulk_training": {
       "enabled": true,
       "max_rows": 100,
       "is_slm_trained_field": "IS_SLM_TRAINED",
       "scenario_id_field": "id"
     },
     "confidence_threshold": 0.7,
     "type_mapping": {
       "int64": "integer",
       "float64": "float",
       "object": "string",
       "datetime64[ns]": "date",
       "timestamp": "string",
       "string": "string",
       "int32": "integer",
       "float32": "float"
     }
   }
   ```

4. **`azure_config.json`**:
   ```json
   {
     "endpoint": "https://your-resource-name.openai.azure.com/",
     "api_key": "<your-api-key>",
     "llm_api_version": "2023-10-01-preview",
     "embedding_api_version": "2023-10-01-preview"
   }
   ```

5. **`aws_config.json`**:
   ```json
   {
     "aws_access_key_id": "<your-access-key-id>",
     "aws_secret_access_key": "<your-secret-access-key>",
     "region": "us-east-1"
   }
   ```

6. **`synonym_config.json`**:
   ```json
   {
     "synonym_mode": "dynamic",
     "dynamic_synonym_threshold": 0.6,
     "custom_synonyms": {}
   }
   ```

**Security Note**: Use environment variables or a secret manager in production.

## File Placement

Place files in `C:\Users\varaprasad\Pythonworks\Text2SQl\Datascriber-2\`:
- **Root**:
  - `main.py` (artifact ID `50242d5b-d690-4dd2-b7f9-da4742fa4dd7`)
  - `requirements.txt`
- **Subdirectories**:
  - `cli/interface.py` (artifact ID `6156cb3f-d2c5-453d-8bb1-207c6903bf6c`)
  - `core/orchestrator.py` (artifact ID `3a2fb687-6893-43f1-9b6a-45fdd0b3db23`)
  - `tia/table_identifier.py` (artifact ID `862b1d15-dc91-47aa-b7e5-5ec57bfe91e3`)
  - `proga/prompt_generator.py` (artifact ID `bcb70158-457d-4120-b23c-6e619551b98a`)
  - `opden/data_executor.py` (artifact ID `f93cc85a-a9de-4ce7-97ae-d659e2fc38ab`)
  - `nlp/nlp_processor.py` (artifact ID `98644235-9805-4fab-b8c2-aee24f94e909`)
  - `storage/db_manager.py` (artifact ID `a3aea774-2a68-47f7-abc4-fa2c2089cff5`)
  - `storage/storage_manager.py` (artifact ID `6f8b32dc-ad35-4a52-bd67-1ef213b8235c`)
  - `config/utils.py` (artifact ID `552166cb-0fd7-4b91-8267-7681f89c4a1e`)
  - `config/logging_setup.py` (artifact ID `817620a9-1f48-4018-86d6-36e45d2308bf`)
- **Configuration**:
  - `app-config/db_configurations.json` (artifact ID `a9f8d7e5-de49-4551-bf98-534f3689adcc`)
  - `app-config/llm_config.json` (artifact ID `55be64f5-c6b8-43c0-86ac-37ef3f5b596d`)
  - `app-config/model_config.json` (artifact ID `45bb8ad4-802e-4d04-8207-f10b10171980`)
  - `app-config/azure_config.json` (artifact ID `c831ec89-23ff-426f-8b46-e72e58fc53e0`)
  - `app-config/aws_config.json` (artifact ID `549d460f-6499-41f1-bfcf-250486ec37ae`)
  - `app-config/synonym_config.json` (artifact ID `b9493716-495e-4458-9d5e-8f171304a952`)
- **Documentation**:
  - `docs/README.md`
  - `docs/Test_Scenarios.md`
  - `docs/Datascriber_Functional_Documentation.md`

## Execution

### Data User
1. **Launch CLI**:
   - SQL Server:
     ```bash
     python main.py --datasource bikestores --debug
     ```
   - S3:
     ```bash
     python main.py --datasource salesdata --debug
     ```
   - Omit `--schema` to search all schemas.
2. **Submit NLQs**:
   - Example: “Show orders from 2016”
   - Expected SQL (S3): 
     ```sql
     SELECT * FROM orders WHERE CAST(order_date AS DATE) LIKE '2016%';
     ```
   - Results displayed in CLI and saved to `temp/query_results/`.
3. **Verification**:
   - Check CLI for SQL and results.
   - Review `logs/datascriber.log` for debug details.
   - See `docs/Test_Scenarios.md` for examples.

### Admin User
1. **Validate Metadata**:
   - Delete metadata:
     ```bash
     rm data/bikestores/metadata_data_*.json
     rm data/salesdata/metadata_data_default.json
     ```
   - Submit query: “Show me customer names”.
   - Check `data/<datasource>/` for regenerated metadata.
2. **Review Rejected Queries**:
   - Submit invalid query: “what”.
   - Query SQLite:
     ```sql
     SELECT * FROM rejected_queries;
     ```
3. **Manage Training Data**:
   - Submit query: “Show orders from 2016”.
   - Query SQLite:
     ```sql
     SELECT * FROM training_data WHERE user_query = 'Show orders from 2016';
     ```
4. **Verification**:
   - Confirm metadata files and SQLite entries.
   - Check `logs/datascriber.log`.

## Testing

- **Manual Testing**: Use `docs/Test_Scenarios.md` for updated scenarios, including DuckDB and date handling tests.
- **Verification**:
  - Confirm table identification without `--schema`.
  - Verify SQL queries and results.
  - Check SQLite and logs for rejections.

## Troubleshooting

- **Connection Issues**:
  - **SQL Server**: Verify `bikestores` credentials.
  - **S3**: Test `aws s3 ls s3://bike-stores-bucket/`.
- **DuckDB Errors**:
  - Ensure `duckdb` is installed (`pip show duckdb`).
  - Check `logs/datascriber.log` for IO errors.
- **Date Handling Issues**:
  - Verify `order_date` format in `orders.csv` (e.g., `YYYY-MM-DD`).
  - Check log for `_adjust_sql_for_date_columns` debug messages.
- **Azure Open AI Errors**:
  - Validate `azure_config.json` credentials.
  - Check rate limits in Azure portal.
- **Metadata Issues**:
  - Delete metadata files and rerun CLI.
  - Verify log for generation errors.
- **Logs**:
  - Use `--debug` for verbose logging.
  - Share log excerpts for support.

## Optional Enhancements

- **Batch Mode**: Implement `--mode batch` in `main.py`.
- **Packaging**: Create `setup.py` for distribution.
- **Deployment**: Develop Docker or cloud guides.
- **CI/CD**: Add GitHub Actions workflow.