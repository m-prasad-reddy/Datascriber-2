# Datascriber 1.1 Test Scenarios

This document outlines test scenarios for manually evaluating Datascriber 1.1 functionality as Data User and Admin User roles on SQL Server (`bikestores`) and S3 (`salesdata`) datasources. The scenarios test Text-to-SQL conversion, table identification, query execution, error handling, and administrative tasks, ensuring alignment with requirements for natural language query (NLQ) processing, metadata-driven table identification, and Azure Open AI integration.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Test Scenarios](#test-scenarios)
  - [Data User Scenarios](#data-user-scenarios)
    - [SQL Server (`bikestores`)](#sql-server-bikestores)
    - [S3 (`salesdata`)](#s3-salesdata)
  - [Admin User Scenarios](#admin-user-scenarios)
    - [SQL Server (`bikestores`)](#sql-server-bikestores-admin)
    - [S3 (`salesdata`)](#s3-salesdata-admin)
- [Execution Instructions](#execution-instructions)
- [Notes](#notes)

## Overview

Datascriber 1.1 converts NLQs into SQL queries for SQL Server and S3 datasources, using Azure Open AI (`gpt-4o`, `text-embedding-3-small`) for query generation and table identification. The test scenarios verify:
- Accurate table identification using metadata from `db_configurations.json` without mandatory schema specification.
- SQL query generation and execution for valid NLQs.
- Error handling for invalid queries.
- Metadata generation and training data management for admin tasks.
- Functionality for both SQL Server (`bikestores`, schemas: `sales`, `production`) and S3 (`salesdata`, schema: `default`).

## Prerequisites

- **Environment**:
  - Python 3.8 or higher.
  - Operating System: Windows, Linux, or macOS.
  - Installed dependencies (see `requirements.txt` in `README.md`).
  - spaCy model: `en_core_web_sm`.
- **Datasources**:
  - **SQL Server (`bikestores`)**: Accessible database with schemas `sales` and `production`, containing tables: `sales.orders`, `sales.order_items`, `sales.customers`, `sales.stores`, `sales.staffs`, `production.products`, `production.categories`, `production.brands`, `production.stocks`.
  - **S3 (`salesdata`)**: S3 bucket (`bike-stores-s3-bucket`) with data files (CSV, Parquet, or ORC) mimicking `bikestores` tables (e.g., `products`, `stores`, `orders`) in the `sales` prefix.
- **Configuration Files**:
  - Located in `app-config/` with valid credentials:
    - `db_configurations.json`: Defines `bikestores` and `salesdata`.
    - `llm_config.json`: Configures `gpt-4o` and `text-embedding-3-small`.
    - `model_config.json`: Sets training and type mapping.
    - `azure_config.json`: Azure Open AI credentials.
    - `aws_config.json`: AWS S3 credentials.
    - `synonym_config.json`: Dynamic synonym settings.
  - See `README.md` for configuration examples.
- **Tools**:
  - SQLite client (e.g., DB Browser for SQLite) for admin tasks.
  - Access to `logs/datascriber.log` for debugging.

## Test Scenarios

### Data User Scenarios

These scenarios test NLQ processing, table identification, query generation, and execution as a Data User.

#### SQL Server (`bikestores`)

1. **Product Stock Availability Across Stores**
   - **Query**: “SHOW ME PRODUCTS STOCK AVAILABILITY AT ALL STORES”
   - **Command**: `python main.py --datasource bikestores --debug`
   - **Expected Behavior**:
     - Identifies tables: `production.products`, `production.stocks`, `sales.stores`.
     - Generates SQL: `SELECT p.product_name, s.store_name, st.quantity FROM production.products p JOIN production.stocks st ON p.product_id = st.product_id JOIN sales.stores s ON st.store_id = s.store_id;`
     - Returns results with product names, store names, and quantities.
   - **Verification**:
     - Check CLI for SQL query and result set.
     - Verify `logs/datascriber.log` for table identification across `sales` and `production` schemas.
     - Ensure no schema prompt if `--schema` is omitted.
   - **Tables**: `production.products`, `production.stocks`, `sales.stores`

2. **Product Stock with Store Details**
   - **Query**: “show me products stock availability at all stores with store details”
   - **Command**: `python main.py --datasource bikestores --debug`
   - **Expected Behavior**:
     - Identifies tables: `production.products`, `production.stocks`, `sales.stores`.
     - Generates SQL: `SELECT p.product_name, s.store_name, s.city, s.state, st.quantity FROM production.products p JOIN production.stocks st ON p.product_id = st.product_id JOIN sales.stores s ON st.store_id = s.store_id;`
     - Returns results with product names, store names, city, state, and quantities.
   - **Verification**:
     - Confirm SQL query includes `city` and `state`.
     - Check results match expected columns.
     - Log confirms metadata-driven identification.
   - **Tables**: `production.products`, `production.stocks`, `sales.stores`

3. **Customer Names**
   - **Query**: “Show me customer names”
   - **Command**: `python main.py --datasource bikestores --debug`
   - **Expected Behavior**:
     - Identifies table: `sales.customers`.
     - Generates SQL: `SELECT first_name, last_name FROM sales.customers;`
     - Returns list of customer names.
   - **Verification**:
     - Verify SQL query and name list output.
     - Log shows `sales` schema access without `--schema`.
   - **Table**: `sales.customers`

4. **Production Categories**
   - **Query**: “what production categories are available”
   - **Command**: `python main.py --datasource bikestores --debug`
   - **Expected Behavior**:
     - Identifies table: `production.categories`.
     - Generates SQL: `SELECT category_name FROM production.categories;`
     - Returns list of category names.
   - **Verification**:
     - Check SQL query and category list.
     - Log confirms `production` schema usage.
   - **Table**: `production.categories`

5. **Store Details for Baldwin Bikes**
   - **Query**: “get me store details of Baldwin Bikes”
   - **Command**: `python main.py --datasource bikestores --debug`
   - **Expected Behavior**:
     - Identifies table: `sales.stores`.
     - Generates SQL: `SELECT * FROM sales.stores WHERE store_name = 'Baldwin Bikes';`
     - Returns store details (e.g., `store_id`, `store_name`, `city`).
   - **Verification**:
     - Verify SQL query filters by “Baldwin Bikes”.
     - Check result matches store data.
   - **Table**: `sales.stores`

6. **Total Sales Amount for Baldwin Bikes**
   - **Query**: “total sales amount at storename 'Baldwin Bikes'”
   - **Command**: `python main.py --datasource bikestores --debug`
   - **Expected Behavior**:
     - Identifies tables: `sales.stores`, `sales.orders`, `sales.order_items`.
     - Generates SQL: `SELECT SUM(oi.quantity * oi.list_price * (1 - oi.discount)) AS total_sales FROM sales.stores s JOIN sales.orders o ON s.store_id = o.store_id JOIN sales.order_items oi ON o.order_id = oi.order_id WHERE s.store_name = 'Baldwin Bikes';`
     - Returns total sales amount.
   - **Verification**:
     - Confirm SQL query calculates sum correctly.
     - Verify result is a single numeric value.
     - Log shows multi-table join.
   - **Tables**: `sales.stores`, `sales.orders`, `sales.order_items`

7. **Orders Delivered in 2016**
   - **Query**: “how many orders were delivered in 2016”
   - **Command**: `python main.py --datasource bikestores --debug`
   - **Expected Behavior**:
     - Identifies table: `sales.orders`.
     - Generates SQL: `SELECT COUNT(*) FROM sales.orders WHERE YEAR(order_date) = 2016;`
     - Returns count of orders.
   - **Verification**:
     - Check SQL query uses `YEAR` function.
     - Verify count matches 2016 orders.
   - **Table**: `sales.orders`

8. **Orders Between Dates**
   - **Query**: “orders processed between Jan-15-2016 and Jan-14-2017”
   - **Command**: `python main.py --datasource bikestores --debug`
   - **Expected Behavior**:
     - Identifies table: `sales.orders`.
     - Generates SQL: `SELECT * FROM sales.orders WHERE order_date BETWEEN '2016-01-15' AND '2017-01-14';`
     - Returns orders within date range.
   - **Verification**:
     - Confirm SQL query uses `BETWEEN` clause.
     - Check results include order details.
   - **Table**: `sales.orders`

9. **Staff Count Across Stores**
   - **Query**: “how many employees works for all stores”
   - **Command**: `python main.py --datasource bikestores --debug`
   - **Expected Behavior**:
     - Identifies tables: `sales.staffs`, `sales.stores`.
     - Generates SQL: `SELECT COUNT(*) FROM sales.staffs s JOIN sales.stores st ON s.store_id = st.store_id;`
     - Returns total staff count.
   - **Verification**:
     - Verify SQL query joins tables.
     - Check count is accurate.
   - **Tables**: `sales.staffs`, `sales.stores`

10. **Invalid Query (Vague)**
    - **Query**: “what”
    - **Command**: `python main.py --datasource bikestores --debug`
    - **Expected Behavior**:
      - Rejected with message: “Please enter a meaningful query (e.g., 'show me all stores with store names').”
      - Logged in `rejected_queries` table.
    - **Verification**:
      - Check CLI error message.
      - Verify `logs/datascriber.log` and `data/bikestores/datascriber.db` entry.
    - **Tables**: None (rejected)

11. **Invalid Query (Non-English)**
    - **Query**: “chitti emchestunnav”
    - **Command**: `python main.py --datasource bikestores --debug`
    - **Expected Behavior**:
      - Rejected with message: “Unable to process query. Please try again or reconnect.”
      - Logged in `rejected_queries`.
    - **Verification**:
      - Check CLI error message.
      - Verify log and database entries.
    - **Tables**: None (rejected)

12. **Invalid Query (Numeric)**
    - **Query**: “12345”
    - **Command**: `python main.py --datasource bikestores --debug`
    - **Expected Behavior**:
      - Rejected with message: “Please enter a meaningful query (e.g., 'show me all stores with store names').”
      - Logged in `rejected_queries`.
    - **Verification**:
      - Check CLI error message.
      - Verify log and database entries.
    - **Tables**: None (rejected)

#### S3 (`salesdata`)

1. **Product Stock Availability Across Stores**
   - **Query**: “SHOW ME PRODUCTS STOCK AVAILABILITY AT ALL STORES”
   - **Command**: `python main.py --datasource salesdata --debug`
   - **Expected Behavior**:
     - Identifies tables: `products`, `stocks`, `stores`.
     - Generates SQL: `SELECT p.product_name, s.store_name, st.quantity FROM products p JOIN stocks st ON p.product_id = st.product_id JOIN stores s ON st.store_id = s.store_id;`
     - Returns data from S3 files.
   - **Verification**:
     - Check SQL query and results.
     - Log confirms S3 file access (`data/salesdata/metadata_data_default.json`).
   - **Tables**: `products`, `stocks`, `stores`

2. **Total Sales Amount for Baldwin Bikes**
   - **Query**: “total sales amount at storename 'Baldwin Bikes'”
   - **Command**: `python main.py --datasource salesdata --debug`
   - **Expected Behavior**:
     - Identifies tables: `stores`, `orders`, `order_items`.
     - Generates SQL: `SELECT SUM(oi.quantity * oi.list_price * (1 - oi.discount)) AS total_sales FROM stores s JOIN orders o ON s.store_id = o.store_id JOIN order_items oi ON o.order_id = oi.order_id WHERE s.store_name = 'Baldwin Bikes';`
     - Returns total sales amount.
   - **Verification**:
     - Verify SQL query and numeric result.
     - Log shows S3 file reads.
   - **Tables**: `stores`, `orders`, `order_items`

3. **List Products by Categories and Brands**
   - **Query**: “list all products by categories and brand names”
   - **Command**: `python main.py --datasource salesdata --debug`
   - **Expected Behavior**:
     - Identifies tables: `categories`, `brands`, `products`.
     - Generates SQL: `SELECT c.category_name, b.brand_name, p.product_name FROM categories c JOIN products p ON c.category_id = p.category_id JOIN brands b ON p.brand_id = b.brand_id;`
     - Returns product details.
   - **Verification**:
     - Check SQL query joins and results.
     - Log confirms `default` schema metadata usage.
   - **Tables**: `categories`, `brands`, `products`

4. **Invalid Query (Vague)**
   - **Query**: “store”
   - **Command**: `python main.py --datasource salesdata --debug`
   - **Expected Behavior**:
     - Rejected with message: “Please enter a meaningful query (e.g., 'show me all stores with store names').”
     - Logged in `rejected_queries`.
   - **Verification**:
     - Check CLI error message.
     - Verify log and `data/salesdata/datascriber.db` entry.
   - **Tables**: None (rejected)

### Admin User Scenarios

These scenarios test configuration, metadata management, and rejected query handling.

#### SQL Server (`bikestores`) Admin

1. **Validate Metadata Generation**
   - **Task**: Ensure metadata is generated for `bikestores`.
   - **Command**: `python main.py --datasource bikestores --debug`
   - **Steps**:
     - Delete `data/bikestores/metadata_data_*.json`.
     - Run command and submit query: “Show me customer names”.
   - **Expected Behavior**:
     - `DBManager.validate_metadata` generates `metadata_data_sales.json` and `metadata_data_production.json`.
     - Query processes normally.
   - **Verification**:
     - Check `data/bikestores/` for metadata files.
     - Verify log for metadata generation.
   - **Tables**: All (`sales.*`, `production.*`)

2. **Review Rejected Queries**
   - **Task**: Check rejected queries after invalid inputs.
   - **Command**: `python main.py --datasource bikestores --debug`
   - **Steps**:
     - Submit invalid queries: “what”, “12345”, “chitti emchestunnav”.
     - Query SQLite: `SELECT * FROM rejected_queries;`
   - **Expected Behavior**:
     - `rejected_queries` table in `data/bikestores/datascriber.db` contains entries with query, reason, and timestamp.
   - **Verification**:
     - Use SQLite client to query database.
     - Confirm entries match submitted queries.
   - **Table**: `rejected_queries` (SQLite)

3. **Update Training Data**
   - **Task**: Add training data for a query.
   - **Command**: `python main.py --datasource bikestores --debug`
   - **Steps**:
     - Submit query: “Show me customer names”.
     - Query SQLite: `SELECT * FROM training_data WHERE user_query = 'Show me customer names';`
   - **Expected Behavior**:
     - `DBManager.store_training_data` adds entry with `user_query`, `relevant_sql`, and `is_slm_trained`.
   - **Verification**:
     - Check `training_data` table for entry.
     - Verify log for training data storage.
   - **Table**: `training_data` (SQLite)

#### S3 (`salesdata`) Admin

1. **Validate S3 Metadata Generation**
   - **Task**: Ensure S3 metadata is generated.
   - **Command**: `python main.py --datasource salesdata --debug`
   - **Steps**:
     - Delete `data/salesdata/metadata_data_default.json`.
     - Run command and submit query: “List products by categories and brands”.
   - **Expected Behavior**:
     - `StorageManager.fetch_metadata` generates `metadata_data_default.json`.
     - Query processes normally.
   - **Verification**:
     - Check `data/salesdata/` for metadata file.
     - Verify log for S3 metadata generation.
   - **Tables**: All (`products`, `stores`, etc.)

2. **Review Rejected Queries**
   - **Task**: Check rejected queries for S3.
   - **Command**: `python main.py --datasource salesdata --debug`
   - **Steps**:
     - Submit invalid query: “store”.
     - Query SQLite: `SELECT * FROM rejected_queries;`
   - **Expected Behavior**:
     - Entry added to `rejected_queries` in `data/salesdata/datascriber.db`.
   - **Verification**:
     - Confirm database entry.
     - Check log for rejection details.
   - **Table**: `rejected_queries` (SQLite)

## Execution Instructions

1. **Setup**:
   - **Clone Repository**:
     ```bash
     git clone https://github.com/m-prasad-reddy/Datascriber-2.git
     cd Datascriber
     ```
   - **Create Virtual Environment**:
     ```bash
     python -m venv venv
     source venv/bin/activate  # On Windows: venv\Scripts\activate
     ```
   - **Install Dependencies**:
     ```bash
     pip install -r requirements.txt
     ```
     Ensure `requirements.txt` includes:
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
   - **Download SpaCy Model**:
     ```bash
     python -m spacy download en_core_web_sm
     ```
   - **Configure Files**:
     - Place configuration files in `app-config/` with valid credentials (see `README.md`).
     - Verify `db_configurations.json` includes `bikestores` and `salesdata`.
   - **Verify Datasources**:
     - Ensure SQL Server `bikestores` is running and accessible.
     - Confirm S3 bucket `bike-stores-s3-bucket` contains data files.
   - **Create Directories**:
     ```bash
     mkdir -p app-config data logs cli core tia proga opden nlp storage
     ```

2. **Run Tests**:
   - **Launch CLI**:
     - For each scenario, run the specified command, e.g.:
       ```bash
       python main.py --datasource bikestores --debug
       ```
       or
       ```bash
       python main.py --datasource salesdata --debug
       ```
   - **Submit Queries**:
     - Enter the query in the CLI prompt (e.g., “SHOW ME PRODUCTS STOCK AVAILABILITY AT ALL STORES”).
     - Observe CLI output for SQL query, results, or error messages.
   - **Admin Tasks**:
     - For metadata validation, delete metadata files before running queries.
     - Use a SQLite client to query `data/<datasource>/datascriber.db` for `rejected_queries` or `training_data`.

3. **Verification**:
   - **CLI Output**: Compare SQL queries and results with expected behavior.
   - **Logs**: Check `logs/datascriber.log` for debug details on table identification, query execution, and errors.
   - **Database**: Verify `rejected_queries` and `training_data` tables in `datascriber.db`.
   - **Metadata**: Confirm metadata files (`metadata_data_*.json`) are generated in `data/<datasource>/`.
   - **Table Identification**: Ensure queries work without `--schema`, using metadata from all configured schemas.

4. **Troubleshooting**:
   - If queries fail, check log for errors (e.g., connection issues, invalid metadata).
   - Verify credentials in `azure_config.json` and `aws_config.json`.
   - Ensure S3 files match expected table structure.

## Notes

- **S3 Data Assumption**: The `salesdata` S3 datasource is assumed to contain tables (`products`, `stores`, `orders`, etc.) mirroring `bikestores` structure. If differences exist, adjust expected SQL queries to match S3 file schemas.
- **Table Identification**: Scenarios test metadata-driven table identification without `--schema`, searching all schemas in `db_configurations.json` (e.g., `sales`, `production` for `bikestores`; `default` for `salesdata`).
- **Negative Cases**: Invalid queries (e.g., “what”, “12345”) test `NLPProcessor` and `Interface` error handling, ensuring user feedback and logging.
- **Admin Tasks**: Focus on metadata generation and data management, critical for Admin User role.
- **Performance**: Queries without `--schema` may be slower for multi-schema datasources; logs can help diagnose delays.
- **Logging**: Debug mode (`--debug`) provides detailed logs for troubleshooting.
- **TIA 1.2 Integration**: Table identification uses `tia/table_identifier.py`, leveraging Azure Open AI embeddings for dynamic synonym mapping.
- **Training Data**: Limited to 100 rows per `model_config.json`, stored in SQLite with `IS_SLM_TRAINED` flag.
- **Security**: Ensure credentials are stored securely (e.g., environment variables) and not hardcoded in production.