from azure.ai.ml import MLClient, Input, load_component
from azure.identity import AzureCliCredential, DefaultAzureCredential
from azure.ai.ml.dsl import pipeline
from pathlib import Path
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.sweep import BanditPolicy, Choice
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import ModelType


ml_client = MLClient.from_config(
    credential=DefaultAzureCredential(), file_name=str(Path("..", "config.json"))
)


cluster = "gpu-cluster"

train_test_split = load_component(
    source=str(Path("..", "components", "train_test_split.yml"))
)
model_selection = load_component(
    source=str(Path("..", "components", "model_selection.yml"))
)
evaluate_model = load_component(
    source=str(Path("..", "components", "evaluate_model.yml"))
)


@pipeline(default_compute=cluster)
def tft_pipeline(
    dataset_input_path,
):

    split_step = train_test_split(
        dataset_input_path=dataset_input_path,
    )

    model_selection_step = model_selection(
        train_input_path=split_step.outputs.train_output_path,
        hidden_size=Choice(range(100, 161, 10)),
        attention_head_size=Choice(range(2, 5)),
    )

    # The result of the various runs is logged on the MLFlow server,
    # but only output from the best run is returned by the step.
    hyperparameter_search = model_selection_step.sweep(
        primary_metric="val_MAE",
        goal="maximize",
        sampling_algorithm="bayesian",
        compute=cluster,
    )

    hyperparameter_search.set_limits(
        max_total_trials=2, max_concurrent_trials=1, timeout=36000
    )
    
    hyperparameter_search.early_termination = BanditPolicy(
        slack_factor= 0.1, delay_evaluation = 3, evaluation_interval = 1
    )

    evaluate_step = evaluate_model(
        train_input_path=split_step.outputs.train_output_path,
        test_input_path=split_step.outputs.test_output_path,
        # The `outputs` attribute hyperparameter search contains the
        # output of the run that performed best
        model_input_path=hyperparameter_search.outputs.model_output_path,   
        params_input_path=hyperparameter_search.outputs.params_output_path,
    )

# latest_version = {i.name : i.latest_version for i in ml_client.data.list()}
# dataset = ml_client.data.get("raw_electricity_load", version=latest_version["raw_electricity_load"])
dataset = ml_client.data.get(name="raw_electricity_load", label="latest") # funktioniert nicht??

pipeline_job = tft_pipeline(
    dataset_input_path=Input(
        type=AssetTypes.URI_FILE,
        path=dataset.path,
    ),
)

pipeline_job = ml_client.jobs.create_or_update(
    pipeline_job, experiment_name="tft_experiment"
)



