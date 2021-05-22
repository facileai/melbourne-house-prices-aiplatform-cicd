import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from fastai.tabular.core import TabularPandas, Categorify, FillMissing
from xgboost import XGBRegressor
import fire
import subprocess
import pickle
import sys
import json
from fastai.basics import patch, Path

@patch
def export(self:TabularPandas, job_dir, fname='data-proc.pkl', pickle_protocol=2):
    "Export the contents of `self` without the items"
    old_to = self
    self = self.new_empty()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pickle.dump(self, open(Path('./'+fname), 'wb'), protocol=pickle_protocol)
        self = old_to
    gcs_pkl_path = '{}/{}'.format(job_dir, fname)
   
    subprocess.check_call(['gsutil', 'cp', fname, gcs_pkl_path],
                              stderr=sys.stdout)

def load_artifact(file_path,filename):
    # Copy the artifact file from GCS
    gcs_model_filepath = '{}/{}'.format(file_path, filename)
    print('Download the artifact {}'.format(gcs_model_filepath))
    subprocess.check_call(['gsutil', 'cp', gcs_model_filepath, filename],
                            stderr=sys.stdout)

    with open(filename, 'rb') as file:
        file_content = json.load(file)

    return file_content
        
def load_pandas(fname):
    "Load in a `TabularPandas` object from `fname`"

    res = pickle.load(open(fname, 'rb'))
    return res

def get_data(train_data_pth,valid_data_pth,artifacts_path,job_dir):
    
    train_df = pd.read_csv(train_data_pth, low_memory=False)
    # valid_df = pd.read_csv(valid_data_pth, low_memory=False)

    # train_df = pd.concat([train_df, valid_df])

    features = load_artifact(artifacts_path,'features.txt')
    cont_nn = features['cont']
    cat_nn = features['cat']
    dep_var = features['dep_var']
    
    cols = set(list(train_df.columns.values)).intersection(set(cat_nn+cont_nn+[dep_var]))
    train_df = train_df[cols]
    
    print('Processing the training set ...')
    procs_nn = [Categorify, FillMissing]
    to_nn = TabularPandas(train_df, procs_nn, cat_nn, cont_nn,
                        splits=None, y_names=dep_var)
    
    print('saving the data processor')
    to_nn.export(job_dir=job_dir,fname='data-proc.pkl')

    
    X_train,Y_train = to_nn.train.xs,to_nn.train.y

    return X_train,Y_train

def train_evaluate(training_dataset_path, validation_dataset_path,artifacts_path,job_dir):
    
    X_train,Y_train = get_data(training_dataset_path,validation_dataset_path,artifacts_path,job_dir)

    print('Downloading Hyper parameters')
    params = load_artifact(artifacts_path,'params.txt')
    params['seed'] = 42
    
    model = XGBRegressor(**params)

    model.fit(
        X_train, 
        Y_train, 
        eval_metric="rmse", 
        eval_set=[(X_train, Y_train)], 
        verbose=True, 
        early_stopping_rounds = 20)
    
    # Save the model
 
    model_filename = 'model.pkl'
    with open(model_filename, 'wb') as model_file:
        pickle.dump(model, model_file)
    gcs_model_path = '{}/{}'.format(job_dir, model_filename)
    subprocess.check_call(['gsutil', 'cp', model_filename, gcs_model_path],
                            stderr=sys.stdout)
    print('Saved model in: {}'.format(model_filename))
    
if __name__ == '__main__':
  fire.Fire(train_evaluate)