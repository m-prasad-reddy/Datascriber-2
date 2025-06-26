from diagrams import Diagram, Cluster
from diagrams.azure.compute import AKS
from diagrams.azure.ml import AzureOpenAI
from diagrams.azure.integration import ServiceBus
from diagrams.azure.database import CosmosDb, SQLDatabases
from diagrams.azure.security import KeyVaults
from diagrams.azure.monitor import Monitor
from diagrams.aws.storage import S3
from diagrams.onprem.client import User
from diagrams.programming.framework import Flask
from diagrams.onprem.compute import Server  # For Output
from diagrams.onprem.analytics import Spark  # For Spark SQL Engine
from diagrams.azure.compute import KubernetesServices  # For K8s Pod Launcher

# Diagram configuration
with Diagram(
    "Datascriber Components Architecture",
    show=True,  # Preview during development
    direction="TB",  # Top-to-bottom layout
    outformat="png",
    filename="datascriber_components",
    graph_attr={"bgcolor": "white", "fontsize": "12"}
):
    # User Layer (Generic)
    data_user = User("Data User\nWeb UI, Chat Widget")
    admin_user = User("Admin User\nAdmin Dashboard")

    # Application Layer (Onprem, hosted on Azure AKS)
    with Cluster("Azure Kubernetes Service (AKS)"):
        flask_frontend = Flask("Flask Web App\nJinja2, Bootstrap, Chat Widget")
        flask_backend = Flask("Flask Backend\nUser Mgmt, Task APIs, Chat APIs\nLangChain, LangGraph, Celery")

    # Cloud Services Layer (Azure)
    openai = AzureOpenAI("Azure OpenAI")
    service_bus = ServiceBus("Azure Service Bus\nTask & Query Trigger")
    cosmos_db = CosmosDb("Azure Cosmos DB\nChat History")
    key_vault = KeyVaults("Azure Key Vault")
    monitor = Monitor("Azure Monitor\nApplication Insights")

    # Data Sources Layer (Azure and AWS)
    sql_server = SQLDatabases("SQL Server\nStructured Data")
    s3 = S3("OOS \n S3-compatible\nArchived Data")

    # Executor Jobs Layer (Onprem, hosted on Azure AKS)
    with Cluster("Azure Kubernetes Service (AKS)"):
        pod_launcher = KubernetesServices("K8s Pod Launcher")
        with Cluster("Spark-on-Kubernetes Cluster\n(Ephemeral)"):
            spark_sql = Spark("Spark SQL Engine")

    # Output (Generic)
    output = Server("Output\nResult File (S3) or Streamed Response")

    # Connections
    data_user >> flask_frontend  # HTTPS
    admin_user >> flask_frontend  # HTTPS
    flask_frontend >> flask_backend  # Internal
    flask_backend >> openai  # REST API
    flask_backend >> sql_server  # Data Access
    flask_backend >> s3  # Data Access
    flask_backend >> service_bus  # Task Queue
    flask_backend >> cosmos_db  # Chat History
    flask_backend >> key_vault  # Secrets
    flask_frontend >> monitor  # Monitoring
    flask_backend >> monitor  # Monitoring
    cosmos_db >> monitor  # Monitoring
    service_bus >> pod_launcher  # Job Trigger
    pod_launcher >> spark_sql  # Pod Launch
    spark_sql >> s3  # Data Access
    spark_sql >> output  # Output
    output >> s3  # Result File
    output >> flask_backend  # Streamed Response
    spark_sql >> key_vault  # Secrets
    spark_sql >> monitor  # Monitoring