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

with Diagram("Datascriber Architecture", show=False, direction="TB"):
    # Users
    data_user = User("Data User\nWeb UI, Chat Widget")
    admin_user = User("Admin User\nAdmin Dashboard")

    # Application Layer
    with Cluster("Azure Kubernetes Service (AKS)"):
        flask_frontend = Flask("Flask Web App\nJinja2, Bootstrap, Chat Widget")
        flask_backend = Flask("Flask Backend\nUser Mgmt, Task APIs, Chat APIs\nLangChain, LangGraph, Celery")

    # Cloud Services
    openai = AzureOpenAI("Azure OpenAI")
    service_bus = ServiceBus("Azure Service Bus\nTask Queue")
    cosmos_db = CosmosDb("Azure Cosmos DB\nChat History")
    key_vault = KeyVaults("Azure Key Vault")
    monitor = Monitor("Azure Monitor\nApplication Insights")

    # Data Sources
    sql_server = SQLDatabases("SQL Server\nStructured Data")
    s3 = S3("AWS S3\nUnstructured Data")

    # Executor Jobs
    with Cluster("Azure Kubernetes Service (AKS)"):
        executor_jobs = AKS("Data Executor Jobs\nContainers, Auto-scaling\nLangChain Tools")

    # Connections
    data_user >> flask_frontend
    admin_user >> flask_frontend
    flask_frontend >> flask_backend
    flask_backend >> openai
    flask_backend >> sql_server
    flask_backend >> s3
    flask_backend >> service_bus
    service_bus >> executor_jobs
    flask_backend >> cosmos_db
    flask_backend >> key_vault
    flask_frontend >> monitor
    flask_backend >> monitor
    cosmos_db >> monitor
    executor_jobs >> monitor
    executor_jobs >> key_vault
