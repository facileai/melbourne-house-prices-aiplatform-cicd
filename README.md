# melbourne-house-prices-aiplatform-cicd

This holds the code used during the webinar MLOPS - From local to production that aired on the 27-May-2021

This code will not run since it requires to have set up your AI platform instance and configure some variables in AzureDevops to allow connection between AzureDevops and GCP. this is not covered here. However a video course that teaches how to do that will be released and this file will be updated accordingly.

Here is a few examples of how the settings look like on AzureDevops.

Variable group:  
- kbf-pipeline-variables

    - AZURE_DEVOPS_BUILD_URL : 'https://dev.azure.com/PROJECT_ID/facileai/_apis/pipelines/PIPELINE_ID/runs'

    - AZURE_DEVOPS_PASSWORD : YOUR_BASE_64_PASSWORD

    - AZURE_DEVOPS_USERNAME : YOUR_AZURE_DEVOPS_USERNAMA

    - BASE_IMAGE_NAME: 'base_image'

    - COMPONENT_URL_SEARCH_PREFIX: 'https://raw.githubusercontent.com/kubeflow/pipelines/0.2.5/components/gcp/'

    - TAG : 'latest'

    - TRAINER_IMAGE_NAME : 'trainer_image'

- dev-kbf-release-variables

    - AZURE_PIPELINE_SOURCE_BRANCH : 'refs/heads/dev'

    - AZURE_PIPELINE_VARIABLE_GROUP_NAME : 'dev-kbf-run-variables'

    - AZURE_PIPELINE_VARIABLE_GROUPS_URL : 'https://dev.azure.com/eliegakuba/facileai/_apis/distributedtask/variablegroups/VARIABLE_GROUP_ID'

    - ENDPOINT: YOUR KUBEFLOW ENDPOINT (e.g: 7e03ed7712a4e302-dot-us-central1.pipelines.googleusercontent.com)