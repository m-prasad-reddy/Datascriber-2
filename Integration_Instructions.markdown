# Datascriber 1.1 Integration Instructions

This document provides detailed instructions for setting up, configuring, and executing Datascriber 1.1, a Text-to-SQL system that converts natural language queries (NLQs) into SQL queries for SQL Server (`bikestores`) and S3 (`salesdata`) datasources. It supports Data User and Admin User roles, leveraging Azure Open AI (`gpt-4o`, `text-embedding-3-small`), bulk training (up to 100 rows with `IS_SLM_TRAINED` flag), and TIA 1.2 for table identification. These instructions ensure successful integration for querying data and managing system configurations.

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
  - Operating System: Windows, Linux, or macOS.
  - Git for cloning the repository.
  - SQLite client (e.g., DB Browser for SQLite) for Admin User tasks.
- **Accounts**:
  - Azure Open AI account with `gpt-4o` and `text-embedding-3-small` deployments.
  - AWS account with S3 access to the `bike-stores-s3-bucket` bucket.
- **Datasources**:
  - **SQL Server (`bikestores`)**: Running instance with `sales` and `production` schemas, containing tables: `orders`, `order_items`, `customers`, `stores`, `staffs`, `products`, `categories`, `brands`, `stocks`.
  - **S3 (`salesdata`)**: S3 bucket with data files (CSV, Parquet, or ORC) mirroring `bikestores` tables (e.g., `products`, `stores`) in the `sales` prefix.

## Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/m-prasad-reddy/Datascriber.git
   cd Datascriber
   ```

2. **Create Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Create `requirements.txt`**:
   Save the following to `requirements.txt` in the root directory:
   ```
   spacy==3.7.2
   openai==1.30.0
   numpy==1.24.3
   pandas==2.0.3
   pyodbc==5.0.1
   boto3==1.34.0
   pyarrow==14.0.1
   python-dotenv==1.0.0
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

Place the following configuration files in the `app-config/` directory, replacing placeholders with valid credentials. Ensure secure storage of credentials (e.g., using environment variables or a secret manager in production).

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
           "bucket_name": "bike-stores-s3-bucket",
           "database": "sales",
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
       "system_prompt": "You are a SQL expert generating valid SQL queries for SQL Server and S3 datasources. Use strftime for dates, LOWER() and LIKE for strings, SUM() and AVG() for numerics. Ensure queries are optimized and safe.",
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
     "aws_secret_access_key": "<your-secret-access-key>"
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

**Security Note**: Replace `<your-secure-password>`, `<your-api-key>`, `<your-access-key-id>`, and `<your-secret-access-key>` with actual credentials. Avoid hardcoding credentials in production environments.

## File Placement

Ensure all components and documentation files are placed in the correct directories under `C:\Users\varaprasad\Pythonworks\Text2SQl\Datascriber\` (or your project root):

- **Root Directory**:
  - `main.py` (artifact ID `50242d5b-d690-4dd2-b7f9-da4742fa4dd7`)
  - `requirements.txt`
- **Subdirectories**:
  - `cli/interface.py` (artifact ID `6156cb3f-d2c5-453d-8bb1-207c6903bf6c`)
  - `core/orchestrator.py` (artifact ID `3a2fb687-6893-43f1-9b6a-45fdd0b3db23`)
  - `tia/table_identifier.py` (artifact ID `862b1d15-dc91-47aa-b7e5-5ec57bfe91e3`)
  - `proga/prompt_generator.py` (artifact ID `bcb70158-457d-4120-b23c-6e619551b98a`)
  - `opden/data_executor.py` (artifact ID `01f0248a-c10f-47b6-b5d1-1d1b104f96e6`)
  - `nlp/nlp_processor.py` (artifact ID `98644235-9805-4fab-b8c2-aee24f94e909`)
  - `storage/db_manager.py` (artifact ID `a3aea774-2a68-47f7-abc4-fa2c2089cff5`)
  - `storage/storage_manager.py` (artifact ID `6f8b32dc-ad35-4a52-bd67-1ef213b8235c`)
  - `config/utils.py` (artifact ID `552166cb-0fd7-4b91-8267-7681f89c4a1e`)
  - `config/logging_setup.py` (artifact ID `817620a9-1f48-4018-86d6-36e45d2308bf`)
- **Configuration Files**:
  - `app-config/db_configurations.json` (artifact ID `a9f8d7e5-de49-4551-bf98-534f3689adcc`)
  - `app-config/llm_config.json` (artifact ID `55be64f5-c6b8-43c0-86ac-37ef3f5b596d`)
  - `app-config/model_config.json` (artifact ID `45bb8ad4-802e-4d04-8207-f10b10171980`)
  - `app-config/azure_config.json` (artifact ID `c831ec89-23ff-426f-8b46-e72e58fc53e0`)
  - `app-config/aws_config.json` (artifact ID `549d460f-6499-41f1-bfcf-250486ec37ae`)
  - `app-config/synonym_config.json` (artifact ID `b9493716-495e-4458-9d5e-8f171304a952`)
- **Documentation**:
  - `docs/README.md` (artifact ID `8308abaf-1dc1-497e-8d9a-3313f9161dd9`)
  - `docs/Test_Scenarios.md` (artifact ID `342f838c-ec79-48b7-8ed7-a92bf96fe5d3`)
  - `docs/Datascriber_Functional_Documentation.md` (artifact ID `36bcf4c8-ae15-4a36-8e3b-3e25eef00e26`)

## Execution

### Data User
1. **Launch CLI**:
   - For SQL Server:
     ```bash
     python main.py --datasource bikestores --debug
     ```
   - For S3:
     ```bash
     python main.py --datasource salesdata --debug
     ```
   - Note: The `--schema` argument is optional; omitting it allows table identification across all schemas in `db_configurations.json` (e.g., `sales`, `production` for `bikestores`; `default` for `salesdata`).
2. **Submit NLQs**:
   - Example query: “Show me products stock availability at all stores”
   - Expected SQL (for `bikestores`): 
     ```sql
     SELECT p.product_name, s.store_name, st.quantity 
     FROM production.products p 
     JOIN production.stocks st ON p.product_id = st.product_id 
     JOIN sales.stores s ON st.store_id = s.store_id;
     ```
   - Expected SQL (for `salesdata`): 
     ```sql
     SELECT p.product_name, s.store_name, st.quantity 
     FROM products p 
     JOIN stocks st ON p.product_id = st.product_id 
     JOIN stores s ON st.store_id = s.store_id;
     ```
   - Results are displayed in the CLI.
3. **Verification**:
   - Check CLI output for the generated SQL query and result set.
   - Review `logs/datascriber.log` for debug details on table identification, query generation, and execution.
   - Refer to `docs/Test_Scenarios.md` for additional query examples (e.g., “total sales amount at storename 'Baldwin Bikes'”).

### Admin User
1. **Validate Metadata**:
   - Delete metadata files to test regeneration:
     ```bash
     rm data/bikestores/metadata_data_*.json
     rm data/salesdata/metadata_data_default.json
     ```
   - Run CLI and submit a query (e.g., “Show me customer names”).
   - Check `data/bikestores/` or `data/salesdata/` for regenerated `metadata_data_*.json` files.
2. **Review Rejected Queries**:
   - Submit an invalid query (e.g., “what”).
   - Use a SQLite client to query:
     ```sql
     SELECT * FROM rejected_queries;
     ```
     in `data/bikestores/datascriber.db` or `data/salesdata/datascriber.db`.
3. **Manage Training Data**:
   - Submit a valid query (e.g., “Show me customer names”).
   - Query SQLite:
     ```sql
     SELECT * FROM training_data WHERE user_query = 'Show me customer names';
     ```
     to verify entry with `user_query`, `relevant_sql`, and `is_slm_trained`.
4. **Verification**:
   - Confirm metadata files exist and contain table/column details.
   - Verify `rejected_queries` and `training_data` tables in `datascriber.db` have expected entries.
   - Check `logs/datascriber.log` for metadata generation and storage events.

## Testing

- **Manual Testing**: Use `docs/Test_Scenarios.md` for 12 Data User scenarios (9 valid, 3 invalid) and 3 Admin User scenarios per datasource (SQL Server and S3).
- **Verification Steps**:
  - Ensure table identification works without `--schema`, leveraging metadata from `db_configurations.json`.
  - Confirm generated SQL queries match expected outputs (e.g., multi-table joins, filters, aggregations).
  - Verify rejected queries are logged in SQLite (`rejected_queries` table) and `datascriber.log`.
  - Check metadata regeneration and training data storage for Admin User tasks.
- **Example Scenarios**:
  - Data User: “Show me products stock availability at all stores” (tests multi-table joins).
  - Data User: “what” (tests error handling for vague queries).
  - Admin User: Delete metadata, submit query, verify regeneration.

## Troubleshooting

- **Connection Issues**:
  - **SQL Server**: Ensure the `bikestores` database is running and credentials in `db_configurations.json` are correct. Check server logs for connection errors.
  - **S3**: Verify AWS credentials in `aws_config.json` and access to `bike-stores-s3-bucket`. Test bucket access with `aws s3 ls s3://bike-stores-s3-bucket/`.
- **Azure Open AI Errors**:
  - Confirm the endpoint and API key in `azure_config.json` are valid.
  - Ensure `gpt-4o` and `text-embedding-3-small` deployments are active in your Azure Open AI account.
  - Check rate limits or network issues in Azure portal.
- **Metadata Issues**:
  - If queries fail, delete `data/<datasource>/metadata_data_*.json` and rerun CLI to regenerate metadata.
  - Verify `logs/datascriber.log` for metadata generation errors (e.g., missing tables, schema access issues).
- **Query Processing Errors**:
  - For invalid SQL or execution failures, check CLI error messages and logs for details.
  - Ensure NLQs are clear and match expected patterns (e.g., avoid vague terms like “what”).
- **Logs**:
  - Review `logs/datascriber.log` for detailed error messages, including stack traces for debugging.
  - Enable `--debug` for verbose logging if issues persist.
- **Support**:
  - Refer to `docs/Datascriber_Functional_Documentation.md` for component-level details and architecture.
  - Share log excerpts or error messages for further assistance.

## Optional Enhancements

The following features are not implemented but can be added to enhance Datascriber:
- **Batch Mode**: Implement `--mode batch` in `main.py` to support bulk query processing.
- **Packaging**: Create a `setup.py` for distributing Datascriber as a Python package.
- **Deployment**: Develop a Docker container or cloud deployment guide (e.g., AWS ECS, Azure App Service).
- **CI/CD**: Configure a GitHub Actions workflow for automated builds and testing.