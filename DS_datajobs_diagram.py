from diagrams import Diagram, Cluster
from diagrams.azure.compute import AKS
from diagrams.azure.integration import ServiceBus
from diagrams.aws.storage import S3
from diagrams.onprem.client import User
from diagrams.programming.framework import Flask
from diagrams.onprem.compute import Server  # For Spark SQL Engine
from diagrams.onprem.analytics import Spark  # For Spark-on-Kubernetes

# Diagram configuration
with Diagram(
    "Datascriber DataExecutor Flow",
    show=True,  # Preview during development
    direction="TB",  # Top-to-bottom layout
    outformat="png",
    filename="datascriber_dataexecutor_flow",
    graph_attr={"bgcolor": "white", "fontsize": "12"}
):
    # User Layer
    data_user = User("Data User\nWeb UI, Chat Widget")

    # Application Layer
    with Cluster("Azure Kubernetes Service (AKS)"):
        flask_backend = Flask("Flask Backend\nTask APIs, LangChain, Celery")

    # Job Trigger
    service_bus = ServiceBus("Azure Service Bus\nDataExecutor\n Job Trigger")

    # Kubernetes Layer
    with Cluster("Azure Kubernetes Service (AKS)"):
        pod_launcher = AKS("K8s Pod Launcher")
        with Cluster("Spark-on-Kubernetes Cluster\n(Ephemeral)"):
            spark_sql = Spark("Spark SQL Engine")

    # Data Source
    s3 = S3("OOS (S3-compatible)\nUnstructured Data")

    # Output
    output = Server("Output\nResult File (S3) \n or Streamed Response")

    # Connections
    data_user >> flask_backend  # HTTPS: User request
    flask_backend >> service_bus  # Task Queue: Job trigger
    service_bus >> pod_launcher  # Job Trigger
    pod_launcher >> spark_sql  # Launch ephemeral Spark cluster
    spark_sql >> s3  # Data Access: Read from S3
    spark_sql >> output  # Output: File to S3 or streamed response
    output >> flask_backend  # Streamed response back to Flask (optional)
    output >> s3  # Result file stored in S3 (optional)