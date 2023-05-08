from pathlib import Path
import os

from dotenv import load_dotenv
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import AzureCliCredential, DefaultAzureCredential
from azure.ai.ml import MLClient

load_dotenv(Path("..", ".env"))

account_name = os.environ["SA_NAME"]
container_name = os.environ["SA_CONTAINER_NAME"]

ml_client = MLClient.from_config(
    credential=DefaultAzureCredential(), file_name="config.json"
)

datastore_name = ml_client.datastores.get_default().name

training_data = Data(
    path=f"azureml://datastores/{datastore_name}/paths/tft_data/training/LD2011_2014.txt",
    type=AssetTypes.URI_FILE,
    description="ElectricityLoadDiagrams20112014 dataset - power usage (kw) of 370 consumers with 15 min frequency",
    name="raw_electricity_load",
)

ml_client.data.create_or_update(training_data)
