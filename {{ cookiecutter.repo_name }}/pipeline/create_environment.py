from pathlib import Path
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
from azure.identity import AzureCliCredential, DefaultAzureCredential


ml_client = MLClient.from_config(
    credential=DefaultAzureCredential(), file_name="config.json"
)

environment = Environment(
    name="tft_environment",
    image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04",
    conda_file=str(Path("conda.yml").resolve()),
    description="Environment belonging to Azure ML Python SDK v2 tft tutorial",
)

ml_client.environments.create_or_update(environment)
