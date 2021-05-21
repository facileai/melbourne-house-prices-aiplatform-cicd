import os

from helper_components import evaluate_model, run_azure_pipeline
import kfp
from kfp.components import func_to_container_op
from kfp.gcp import use_gcp_secret

# Defaults and environment settings
BASE_IMAGE = os.getenv('BASE_IMAGE')
TRAINER_IMAGE = os.getenv('TRAINER_IMAGE')
RUNTIME_VERSION = os.getenv('RUNTIME_VERSION')
PYTHON_VERSION = os.getenv('PYTHON_VERSION')
COMPONENT_URL_SEARCH_PREFIX = os.getenv('COMPONENT_URL_SEARCH_PREFIX')
USE_KFP_SA = os.getenv('USE_KFP_SA')
if USE_KFP_SA == 'True':
    USE_KFP_SA = True
else:
    USE_KFP_SA = False

AZURE_DEVOPS_USERNAME = os.getenv('AZURE_DEVOPS_USERNAME')
AZURE_DEVOPS_PASSWORD = os.getenv('AZURE_DEVOPS_PASSWORD')
AZURE_DEVOPS_BUILD_URL = os.getenv('AZURE_DEVOPS_BUILD_URL')
AZURE_PIPELINE_SOURCE_BRANCH = os.getenv('AZURE_PIPELINE_SOURCE_BRANCH')
AZURE_PIPELINE_VARIABLE_GROUPS_URL = os.getenv('AZURE_PIPELINE_VARIABLE_GROUPS_URL')
AZURE_PIPELINE_VARIABLE_GROUP_NAME = os.getenv('AZURE_PIPELINE_VARIABLE_GROUP_NAME')

TRAINING_FILE_PATH = 'datasets/training/data.csv'
VALIDATION_FILE_PATH = 'datasets/validation/data.csv'
TESTING_FILE_PATH = 'datasets/testing/data.csv'


# Create component factories
component_store = kfp.components.ComponentStore(
        local_search_paths=None, url_search_prefixes=[COMPONENT_URL_SEARCH_PREFIX])

mlengine_train_op = component_store.load_component('ml_engine/train')
evaluate_model_op = func_to_container_op(evaluate_model, base_image=BASE_IMAGE)
run_azure_pipeline_op = func_to_container_op(run_azure_pipeline, base_image=BASE_IMAGE)
    
@kfp.dsl.pipeline(
    name='Predict Melbourne House Pricing - Training',
    description='The pipeline trains and deploys a model that predicts House prices from a Melbourne House pricing list dataset released by Kaggle'
)
def covid_deaths_train(project_id,
                    region,
                    gcs_root,
                    evaluation_metric_name,
                    evaluation_metric_threshold,
                    ):
    """Orchestrates training and deployment of an xgboost model with fastai data preprocessor."""

    # data  paths

    training_file_path = '{}/{}'.format(gcs_root, TRAINING_FILE_PATH)

    validation_file_path = '{}/{}'.format(gcs_root, VALIDATION_FILE_PATH)

    testing_file_path = '{}/{}'.format(gcs_root, TESTING_FILE_PATH)


    # Train the model on a combined training and validation datasets
    job_dir = '{}/{}/{}'.format(gcs_root, 'jobdir', kfp.dsl.RUN_ID_PLACEHOLDER)
    artifacts_dir = '{}/{}'.format(gcs_root, 'artifacts')
   
    train_args = [
        '--training_dataset_path',
        training_file_path,
        '--validation_dataset_path',
        validation_file_path,
        '--artifacts_path',
        artifacts_dir
    ]

    train_model = mlengine_train_op(
        project_id=project_id,
        region=region,
        master_image_uri=TRAINER_IMAGE,
        job_dir=job_dir,
        args=train_args)

    # Evaluate the model on the testing split
    eval_model = evaluate_model_op(
        dataset_path=testing_file_path,
        model_path=str(train_model.outputs['job_dir']),
        metric_name=evaluation_metric_name)

    # Trigger AzureDevops Pipeline to deploy the model if the primary metric is better than threshold
    with kfp.dsl.Condition(eval_model.outputs['metric_value'] < evaluation_metric_threshold):

          run_azure_pipeline = run_azure_pipeline_op(
             username=AZURE_DEVOPS_USERNAME,
             password=AZURE_DEVOPS_PASSWORD,
             build_queue_url=AZURE_DEVOPS_BUILD_URL,
             source_branch=AZURE_PIPELINE_SOURCE_BRANCH,
             variable_groups_url=AZURE_PIPELINE_VARIABLE_GROUPS_URL,
             variable_group_name=AZURE_PIPELINE_VARIABLE_GROUP_NAME,
             run_id=kfp.dsl.RUN_ID_PLACEHOLDER)

    # Configure the pipeline to run using the service account defined
    # in the user-gcp-sa k8s secret
    if USE_KFP_SA == 'True':
        kfp.dsl.get_pipeline_conf().add_op_transformer(
              use_gcp_secret('user-gcp-sa'))