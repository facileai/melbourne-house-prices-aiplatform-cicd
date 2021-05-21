"""Helper components."""

from typing import NamedTuple

def evaluate_model(
    dataset_path: str, model_path: str, metric_name: str
) -> NamedTuple('Outputs', [('metric_name', str), ('metric_value', float),('mlpipeline_metrics', 'Metrics')]):
  """Evaluates a trained xgboost model."""
  #import joblib
  import pickle
  import json
  import pandas as pd
  import subprocess
  import sys
  import numpy as np
  from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
  
  print('Download the dataset {}'.format(dataset_path))
  df_test = pd.read_csv(dataset_path,low_memory=False)

  # Copy the model from GCS
  model_filename = 'model.pkl'
  gcs_model_filepath = '{}/{}'.format(model_path, model_filename)
  print('Download the model {}'.format(gcs_model_filepath))
  subprocess.check_call(['gsutil', 'cp', gcs_model_filepath, model_filename],
                        stderr=sys.stdout)
  print('Model copied from {}'.format(gcs_model_filepath))
  with open(model_filename, 'rb') as model_file:
    model = pickle.load(model_file)
    
  # Copy the data processor file from GCS
  preproc_filename = 'data-proc.pkl'
  gcs_model_filepath = '{}/{}'.format(model_path, preproc_filename)
  print('Download the procs {}'.format(gcs_model_filepath))
  subprocess.check_call(['gsutil', 'cp', gcs_model_filepath, preproc_filename],
                        stderr=sys.stdout)

  with open(preproc_filename, 'rb') as preproc_file:
    preproc = pickle.load(preproc_file)
    
  print('Preprocess the test dataset...')
  test_preproc = preproc.train.new(df_test)
  test_preproc.process()
  X_test,y_test = test_preproc.train.xs,test_preproc.train.y
    
  print('Predict from the test set.')
  y_hat = model.predict(X_test)

  if metric_name == 'mse':
    metric_value = mean_squared_error(y_test, y_hat)
  elif metric_name == 'rmse':
    metric_value = np.sqrt(mean_squared_error(y_test, y_hat))
  elif metric_name == 'r2_score':
    metric_value = r2_score(y_test, y_hat)
  else:
    metric_name = 'N/A'
    metric_value = 0

  # Export the metric
  metrics = {
      'metrics': [{
          'name': metric_name,
          'numberValue': float(metric_value)
      }]
  }

  return (metric_name, metric_value, json.dumps(metrics))

def run_azure_pipeline(username: str, password: str, build_queue_url: str, source_branch: str, 
variable_groups_url: str,variable_group_name: str, run_id: str) -> int:
    """Returns 0 if success otherwise non zero number"""
    import requests
    from requests.exceptions import HTTPError
    from requests.auth import HTTPBasicAuth

    try:
        response = requests.put(
            url=variable_groups_url,
            params={'api-version': '5.1-preview.1'},
            headers={'Content-Type': 'application/json'},
            auth = HTTPBasicAuth(username, password),
            json = {'type': 'Vsts','name':variable_group_name,'variables': {'kbf.run.id':{'isSecret':False,'value':run_id}}}
        )
        response.raise_for_status()
    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')  # Python 3.6
        raise
    except Exception as err:
        print(f'Other error occurred: {err}')  # Python 3.6  
        raise
    finally:
      print(f'Updated {variable_group_name} with {run_id}')  
    
    try:
        response = requests.post(
            url=build_queue_url,
            params={'api-version': '6.0-preview.1'},
            headers={'Content-Type': 'application/json'},
            auth = HTTPBasicAuth(username, password),
            json = { 'stagesToSkip':[],
                    'resources':{
                      'repositories': {
                        'self': {
                          'refName': source_branch
                        }
                      }
                    },
                    'templateParameters': {
                      'kfpTrigger': source_branch
                    },
                    'variables': {}
                  }
        )
        
        response.raise_for_status()
    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')  # Python 3.6
        raise
    except Exception as err:
        print(f'Other error occurred: {err}')  # Python 3.6
        raise
    finally:
        print(response.status_code)
        if response.status_code != 200:
            raise Exception(f'status code {response.status_code}')
        else :
            return response.status_code