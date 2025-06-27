import argparse
import json
import math
from jinja2 import Environment, FileSystemLoader


def read_and_round_table_sizes(file_path):
    with open(file_path, 'r') as f:
        raw_sizes = json.load(f)
    rounded = {path: math.ceil(size) for path, size in raw_sizes.items()}
    return rounded


def estimate_cluster_config(rounded_sizes, shuffle_factor=2.0):
    total_input_gb = sum(rounded_sizes.values())
    estimated_shuffle = total_input_gb * shuffle_factor
    total_gb = total_input_gb + estimated_shuffle

    if total_gb <= 10:
        config = {"executor_memory": "6g", "executor_cores": 2, "num_executors": 3}
    elif total_gb <= 20:
        config = {"executor_memory": "8g", "executor_cores": 3, "num_executors": 4}
    elif total_gb <= 40:
        config = {"executor_memory": "12g", "executor_cores": 3, "num_executors": 6}
    elif total_gb <= 80:
        config = {"executor_memory": "16g", "executor_cores": 4, "num_executors": 8}
    else:
        config = {"executor_memory": "20g", "executor_cores": 4, "num_executors": 10}

    config["driver_memory"] = "2g"
    config["estimated_shuffle_gb"] = round(estimated_shuffle, 1)
    config["estimated_total_gb"] = round(total_gb, 1)

    return config, total_input_gb


def convert_to_table_arg_list(rounded_sizes, default_format="parquet"):
    return [{"name": path.strip('/').split('/')[-1], "path": path, "format": default_format}
            for path in rounded_sizes.keys()]


def render_spark_application_yaml(
    job_name, docker_image, sql_query, output_path, output_format,
    tables, config, s3_access_key, s3_secret_key, s3_endpoint,
    template_path=".", template_file="spark_job_template.yaml.j2"
):
    env = Environment(loader=FileSystemLoader(template_path), trim_blocks=True, lstrip_blocks=True)
    template = env.get_template(template_file)

    rendered = template.render(
        job_name=job_name,
        docker_image=docker_image,
        sql_query=sql_query,
        output_path=output_path,
        output_format=output_format,
        table_args=tables,
        num_executors=config['num_executors'],
        executor_memory=config['executor_memory'],
        executor_cores=config['executor_cores'],
        driver_memory=config['driver_memory'],
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        s3_endpoint=s3_endpoint
    )
    return rendered


def main():
    parser = argparse.ArgumentParser(description="Generate SparkApplication YAML from table stats + SQL")
    parser.add_argument("--sql", required=True, help="SQL query string")
    parser.add_argument("--table-stats", required=True, help="Path to JSON file with table sizes (in GB)")
    parser.add_argument("--output-path", required=True, help="S3 path for query result output")
    parser.add_argument("--output-format", default="parquet", help="Output format: parquet, orc, csv, json")
    parser.add_argument("--job-name", required=True, help="Unique Spark job name")
    parser.add_argument("--docker-image", required=True, help="Docker image with PySpark + S3 support")
    parser.add_argument("--s3-access-key", required=True)
    parser.add_argument("--s3-secret-key", required=True)
    parser.add_argument("--s3-endpoint", required=True)
    parser.add_argument("--template-file", default="spark_job_template.yaml.j2", help="Path to YAML template")
    parser.add_argument("--output-yaml", default="generated_spark_job.yaml", help="Where to save the final YAML")

    args = parser.parse_args()

    # 1. Read and round sizes
    rounded_sizes = read_and_round_table_sizes(args.table_stats)

    # 2. Estimate config
    config, input_gb = estimate_cluster_config(rounded_sizes)

    # 3. Prepare tables list
    table_list = convert_to_table_arg_list(rounded_sizes)

    # 4. Render YAML
    yaml_str = render_spark_application_yaml(
        job_name=args.job_name,
        docker_image=args.docker_image,
        sql_query=args.sql,
        output_path=args.output_path,
        output_format=args.output_format,
        tables=table_list,
        config=config,
        s3_access_key=args.s3_access_key,
        s3_secret_key=args.s3_secret_key,
        s3_endpoint=args.s3_endpoint,
        template_file=args.template_file
    )

    with open(args.output_yaml, "w") as f:
        f.write(yaml_str)

    print(f"✅ YAML generated at: {args.output_yaml}")
    print(f"📦 Estimated input: {input_gb} GB, Shuffle: {config['estimated_shuffle_gb']} GB")
    print(f"🧠 Suggested config: {config['num_executors']} executors, "
          f"{config['executor_memory']} mem, {config['executor_cores']} cores")

if __name__ == "__main__":
    main()
# This code is designed to generate a Spark job configuration YAML file based on input SQL queries and table statistics.
# It reads table sizes from a JSON file, estimates the required cluster configuration, and renders a YAML template using Jinja2.
# The generated YAML can be used to submit Spark jobs with the specified parameters.
# The script also includes command-line argument parsing for flexibility in usage.
# The output YAML file contains all necessary configurations for running the Spark job, including S3 access credentials and job parameters.
# The script is intended to be run from the command line, making it suitable for integration into data processing pipelines or automation scripts.
# It is designed to be flexible and extensible, allowing users to customize the job parameters and output formats as needed.
# The use of Jinja2 templating allows for easy modification of the YAML structure without changing the core logic of the script.
# The script is efficient in estimating the cluster configuration based on input sizes and provides a clear output of the generated YAML file.
# The code is structured to be modular, with separate functions for reading table sizes, estimating configurations, converting table arguments, and rendering the YAML template.
# This modularity enhances readability and maintainability, making it easier to adapt the script for different use cases or to extend its functionality in the future.
# The script is designed to be run in a Python environment with the necessary dependencies installed, such as Jinja2 for templating and argparse for command-line argument parsing.
# It is suitable for users familiar with Spark, data processing, and YAML configurations, providing a powerful tool for automating Spark job submissions.
# The script can be easily integrated into larger data processing workflows, making it a valuable asset for data engineers and data scientists working with Spark and large datasets.
# The generated YAML file can be used directly with Spark job submission tools, streamlining the process of running complex data processing tasks in a distributed environment.
# The script is designed to be user-friendly, with clear command-line options and helpful output messages.
#---------- Usage ----------------------------------------
#python generate_spark_job.py 
#  --sql "SELECT order_id,customer_id,order_status,required_date,shipped_date,store_id,staff_id WHERE YEAR(order_date)=2016" 
#  --table-stats table_sizes.json 
#  --output-path s3://bike-stores-bucket/data-files/outgoing/ 
#  --output-format parquet 
#  --job-name order-query 
#  --docker-image yourdockeruser/spark-s3:latest 
#  --s3-access-key DSJSDISDDJKSDDKASDADK 
#  --s3-secret-key AGSSGWUIWJWUWUWUWDHJVADSJBHEFCAAJJE
#  --s3-endpoint https://bike-stores-bucket.s3.dualstack.us-east-1.amazonaws.com 
#  --output-yaml my_spark_app.yaml