# Datascriber 2.0 Test Scenarios

This document outlines test scenarios for evaluating Datascriber 2.0 functionality as Data User and Admin User roles on SQL Server (`bikestores`) and S3 (`salesdata`) datasources. Scenarios test Text-to-SQL conversion, table identification, query execution, error handling, and admin tasks, with new tests for DuckDB integration and date handling.

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

Datascriber 2.0 enhances NLQ processing with DuckDB for S3 queries and improved date handling. Tests verify:
- Table identification without mandatory `--schema`.
- SQL generation and execution, including date queries.
- Error handling for invalid queries.
- Metadata generation and training data management.
- Datasources: SQL Server (`bikestores`, schemas: `sales`, `production`) and S3 (`salesdata`, schema: `default`).

## Prerequisites

- **Environment**:
  - Python 3.8 or higher.
  - OS: Windows, Linux, or macOS.
  - Dependencies per `requirements.txt`.
  - spaCy model: `en_core_web_sm`.
- **Datasources**:
  - **SQL Server**: `bikestores` with `sales` and `production` schemas.
  - **S3**: `bike-stores-bucket` with CSV files in `data-files/`.
- **Configuration**:
  - `app-config/` with valid credentials (see `README.md`).
- **Tools**:
  - SQLite client.
  - Access to `logs/datascriber.log`.

## Test Scenarios

### Data User Scenarios

#### SQL Server (`bikestores`)

1. **Product Stock Availability**
   - **Query**: “SHOW ME PRODUCTS STOCK AVAILABILITY AT ALL STORES”
   - **Command**: `python main.py --datasource bikestores --debug`
   - **Expected**:
     - Tables: `production.products`, `production.stocks`, `sales.stores`.
     - SQL: `SELECT p.product_name, s.store_name, st.quantity FROM production.products p JOIN production.stocks st ON p.product_id = st.product_id JOIN sales.stores s ON st.store_id = s.store_id;`
     - Results: Product names, store names, quantities.
   - **Verification**:
     - Check CLI output and `logs/datascriber.log`.
     - No schema prompt without `--schema`.
   - **Tables**: `production.products`, `production.stocks`, `sales.stores`

2. **Customer Names**
   - **Query**: “Show me customer names”
   - **Command**: `python main.py --datasource bikestores --debug`
   - **Expected**:
     - Table: `sales.customers`.
     - SQL: `SELECT first_name, last_name FROM sales.customers;`
     - Results: Customer names.
   - **Verification**:
     - Check CLI and log.
   - **Table**: `sales.customers`

3. **Orders from 2016**
   - **Query**: “Show orders from 2016”
   - **Command**: `python main.py --datasource bikestores --debug`
   - **Expected**:
     - Table: `sales.orders`.
     - SQL: `SELECT * FROM sales.orders WHERE YEAR(order_date) = 2016;`
     - Results: Orders from 2016.
   - **Verification**:
     - Check CLI, log, and `temp/query_results/output_*.csv`.
   - **Table**: `sales.orders`

4. **Invalid Query (Vague)**
   - **Query**: “what”
   - **Command**: `python main.py --datasource bikestores --debug`
   - **Expected**:
     - Rejected with message: “Please enter a meaningful query.”
     - Logged in `rejected_queries`.
   - **Verification**:
     - Check CLI, log, and SQLite.
   - **Tables**: None

#### S3 (`salesdata`)

1. **Product Stock Availability**
   - **Query**: “SHOW ME PRODUCTS STOCK AVAILABILITY AT ALL STORES”
   - **Command**: `python main.py --datasource salesdata --debug`
   - **Expected**:
     - Tables: `products`, `stocks`, `stores`.
     - SQL: `SELECT p.product_name, s.store_name, st.quantity FROM products p JOIN stocks st ON p.product_id = st.product_id JOIN stores s ON st.store_id = s.store_id;`
     - Results: Data from S3 via DuckDB.
   - **Verification**:
     - Check CLI, log, and `temp/query_results/`.
   - **Tables**: `products`, `stocks`, `stores`

2. **Orders from 2016**
   - **Query**: “Show orders from 2016”
   - **Command**: `python main.py --datasource salesdata --debug`
   - **Expected**:
     - Table: `orders`.
     - SQL: `SELECT * FROM orders WHERE CAST(order_date AS DATE) LIKE '2016%';`
     - Results: Orders from 2016.
   - **Verification**:
     - Check CLI, log, and CSV output.
   - **Table**: `orders`

3. **Invalid Query (Vague)**
   - **Query**: “store”
   - **Command**: `python main.py --datasource salesdata --debug`
   - **Expected**:
     - Rejected with message: “Please enter a meaningful query.”
     - Logged in `rejected_queries`.
   - **Verification**:
     - Check CLI, log, and SQLite.
   - **Tables**: None

### Admin User Scenarios

#### SQL Server (`bikestores`) Admin

1. **Metadata Generation**
   - **Task**: Validate metadata generation.
   - **Command**: `python main.py --datasource bikestores --debug`
   - **Steps**:
     - Delete `data/bikestores/metadata_data_*.json`.
     - Submit query: “Show me customer names”.
   - **Expected**:
     - Generates `metadata_data_sales.json`, `metadata_data_production.json`.
   - **Verification**:
     - Check `data/bikestores/` and log.
   - **Tables**: All

2. **Review Rejected Queries**
   - **Task**: Check rejected queries.
   - **Command**: `python main.py --datasource bikestores --debug`
   - **Steps**:
     - Submit: “what”.
     - Query SQLite: `SELECT * FROM rejected_queries;`
   - **Expected**:
     - Entry in `rejected_queries`.
   - **Verification**:
     - Check SQLite and log.
   - **Table**: `rejected_queries`

#### S3 (`salesdata`) Admin

1. **Metadata Generation**
   - **Task**: Validate S3 metadata.
   - **Command**: `python main.py --datasource salesdata --debug`
   - **Steps**:
     - Delete `data/salesdata/metadata_data_default.json`.
     - Submit query: “Show orders from 2016”.
   - **Expected**:
     - Generates `metadata_data_default.json`.
   - **Verification**:
     - Check `data/salesdata/` and log.
   - **Tables**: All

2. **Review Rejected Queries**
   - **Task**: Check rejected queries.
   - **Command**: `python main.py --datasource salesdata --debug`
   - **Steps**:
     - Submit: “store”.
     - Query SQLite: `SELECT * FROM rejected_queries;`
   - **Expected**:
     - Entry in `rejected_queries`.
   - **Verification**:
     - Check SQLite and log.
   - **Table**: `rejected_queries`

## Execution Instructions

1. **Setup**:
   - Clone: `git clone https://github.com/m-prasad-reddy/Datascriber-2.git`
   - Create virtual environment: `python -m venv venv`
   - Install dependencies: `pip install -r requirements.txt`
   - Download SpaCy: `python -m spacy download en_core_web_sm`
   - Configure `app-config/` files.
   - Create directories: `mkdir -p app-config data logs cli core tia proga opden nlp storage`
   - Verify datasources.

2. **Run Tests**:
   - Launch CLI: `python main.py --datasource <name> --debug`
   - Submit queries per scenarios.
   - Admin tasks: Delete metadata, query SQLite.

3. **Verification**:
   - Check CLI output, `logs/datascriber.log`, SQLite, and `temp/query_results/`.
   - Ensure table identification works without `--schema`.

4. **Troubleshooting**:
   - Check credentials and S3 file structure.
   - Verify `duckdb` installation for S3 queries.
   - Delete metadata for regeneration.

## Notes

- **S3 Data**: Assumes CSV files in `s3://bike-stores-bucket/data-files/`.
- **Table Identification**: Works without `--schema` using metadata.
- **DuckDB**: Enhances S3 query performance.
- **Date Handling**: Supports `YYYY-MM-DD` formats in `orders.csv`.
- **Security**: Use environment variables for credentials.