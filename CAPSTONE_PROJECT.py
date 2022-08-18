import os
import glob
import json
import requests as r
from datetime import datetime, timedelta
import pandas as pd
import pickle

# user made py file
import user_define

from sqlalchemy import create_engine

from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash_operator import BashOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.models import Variable
from airflow.operators.s3_file_transform_operator import S3FileTransformOperator



BUCKET_NAME = 'cap-stone-data-lake'

default_args = {
    'owner': 'airflow',
    'start_date': datetime.now(),
    'schedule_interval': '@once'    #just run one time
    
}
  

def read_train_data:
    
    bucket_name=Variable.get('BUCKET_NAME')
    s3 = S3Hook(aws_conn_id='aws_default')
   
    keys = s3.list_keys(bucket_name=Variable.get('BUCKET_NAME'),
                        prefix=f'traindata/',
                        delimiter="/")
    
    for filename in keys:
        filepath = f"s3://{bucket_name}/traindata/{filename}"
        df_train = pd.read_csv(filepath, header=None)
    
    return df_train


def train_predictor(**kwargs):
    """
    1. read df_train from operator "train_predictor".
    2. data waggling and generate predictor in .sav file for 3 types building.
    """
    
    # read df_train from operator "train_predictor"
    ti = kwargs['ti']
    df_train = ti.xcom_pull(task_ids='read_train_data')  
    
    
    # data waggling and generate predictor in .sav file for 3 types building.
    # syntax detailed in user_define.py file
    
    home_bldg_model, office_bldg_model, hotel_bldg_model = user_define.train_data_predictor(df_train)
    
    
    Home_filename = 'home_bldg_model.sav'
    pickle.dump(home_bldg_model, open(Home_filename, 'wb'))
    
    Office_filename = 'office_bldg_model.sav'
    pickle.dump(office_bldg_model, open(Office_filename, 'wb'))
    
    Hotel_filename = 'hotel_bldg_model.sav'
    pickle.dump(hotel_bldg_model, open(Hotel_filename, 'wb'))
    
    s3_hook = S3Hook(aws_conn_id='aws_default')
    
    # upload home bldg model to under home folder
    s3_hook.load_file(
        filename=home_bldg_model,
        key="home/f'{filename}'",       
        bucket_name=BUCKET_NAME,
    )
    
    # upload office bldg model to under office folder
    s3_hook.load_file(
        filename=office_bldg_model,
        key="office/f'{filename}'",       
        bucket_name=BUCKET_NAME,
    )
    
     # upload hotel bldg model to under hotel folder
    s3_hook.load_file(
        filename=hotel_bldg_model,
        key="hotel/f'{filename}'",       
        bucket_name=BUCKET_NAME,
    )
    
    return print("Predictor uploaded to sub-folder...")

def predict_home_test(**kwargs):
    
    """
    
    1. apply predictor on the home_test.csv file
    2. generate home_predict.csv file
    
    """
    bucket_name=Variable.get('BUCKET_NAME')
    s3 = S3Hook(aws_conn_id='aws_default')
   
    model_keys = s3.list_keys(bucket_name=Variable.get('BUCKET_NAME'),
                        prefix=f'home/*.sav',
                        delimiter="/")
    test_keys = s3.list_keys(bucket_name=Variable.get('BUCKET_NAME'),
                        prefix=f'home/*.csv',
                        delimiter="/")
    
    model_filepath = f"s3://{bucket_name}/home/{model_keys}"
    test_filepath = f"s3://{bucket_name}/home/{test_keys}" 
    
    home_model = pickle.load(open(model_filepath, 'rb'))
    df_home_test = pd.read_csv(test_filepath, header=None)
    cols = [x for x in df_home_test.columns]
    cols.remove('Property ID','Property Name')   
    
    energy_predict = home_model.predict(df_home_test[cols])

    df_home_test['energy_predict'] = energy_predict
    
    home_test = df_home_test.to_csv(index=False)
    
    return home_test

def predict_office_test(**kwargs):
    
    """
    
    1. apply predictor on the home_test.csv file
    2. generate home_predict.csv file
    
    """
    bucket_name=Variable.get('BUCKET_NAME')
    s3 = S3Hook(aws_conn_id='aws_default')
   
    model_keys = s3.list_keys(bucket_name=Variable.get('BUCKET_NAME'),
                        prefix=f'office/*.sav',
                        delimiter="/")
    test_keys = s3.list_keys(bucket_name=Variable.get('BUCKET_NAME'),
                        prefix=f'office/*.csv',
                        delimiter="/")
    
    model_filepath = f"s3://{bucket_name}/office/{model_keys}"
    test_filepath = f"s3://{bucket_name}/office/{test_keys}" 
    
    office_model = pickle.load(open(model_filepath, 'rb'))
    df_office_test = pd.read_csv(test_filepath, header=None)
    cols = [x for x in df_office_test.columns]
    cols.remove('Property ID','Property Name')  
    
    energy_predict = office_model.predict(df_office_test[cols])

    df_office_test['energy_predict'] = energy_predict
    
    office_test = df_office_test.to_csv(index=False)
    
    return office_test


def predict_hotel_test(**kwargs):
    
    """
    
    1. apply predictor on the home_test.csv file
    2. generate home_predict.csv file
    
    """
    bucket_name=Variable.get('BUCKET_NAME')
    s3 = S3Hook(aws_conn_id='aws_default')
   
    model_keys = s3.list_keys(bucket_name=Variable.get('BUCKET_NAME'),
                        prefix=f'hotel/*.sav',
                        delimiter="/")
    test_keys = s3.list_keys(bucket_name=Variable.get('BUCKET_NAME'),
                        prefix=f'hotel/*.csv',
                        delimiter="/")
    
    model_filepath = f"s3://{bucket_name}/hotel/{model_keys}"
    test_filepath = f"s3://{bucket_name}/hotel/{test_keys}" 
    
    hotel_model = pickle.load(open(model_filepath, 'rb'))
    df_hotel_test = pd.read_csv(test_filepath, header=None)
    cols = [x for x in df_hotel_test.columns]
    cols.remove('Property ID','Property Name')  
    
    energy_predict = hotel_model.predict(df_hotel_test[cols])

    df_hotel_test['energy_predict'] = energy_predict
    
    hotel_test = df_hotel_test.to_csv(index=False)
    
    return hotel_test

def update_postgre_table(**kwargs):
    
    ti = kwargs['ti']
    home_test = ti.xcom_pull(task_ids='predict_home_test') 
    office_test = ti.xcom_pull(task_ids='predict_office_test')
    hotel_test = ti.xcom_pull(task_ids='predict_hotel_test')
    
    engine = create_engine('postgresql://postgres:1989O2@localhost:5432/capstone_pred')
    
    df_home_result = pd.read_csv('home_test.csv')
    df_home_result.columns = [c.lower() for c in df.columns] # PostgreSQL doesn't like capitals or spaces
    
    df_office_result = pd.read_csv('office_test.csv')
    df_home_result.columns = [c.lower() for c in df.columns] 
    
    df_hotel_result = pd.read_csv('hotel_test.csv')
    df_hotel_result.columns = [c.lower() for c in df.columns] 
 
    df_home_result.to_sql("Prediction_Result",
          engine,
          if_exists="append",  # Options are ‘fail’, ‘replace’, ‘append’, default ‘fail’
          index = False, # Do not output the index of the dataframe
          )
    
    df_office_result.to_sql("Prediction_Result",
          engine,
          if_exists="append",  
          index = False, 
          )
    
    df_hotel_result.to_sql("Prediction_Result",
          engine,
          if_exists="append",  
          index = False, 
          )
    
    return print("Prediction Result table updated...")



def read_from_bucket(foldername):
    ti = kwargs['ti']
    submission_filename = ti.xcom_pull(task_ids='user_submission')
    
    bucket_name=Variable.get('BUCKET_NAME')
    s3 = S3Hook(aws_conn_id='aws_default')
   
    keys = s3.list_keys(bucket_name=Variable.get('BUCKET_NAME'),
                        prefix=f'/{foldername}/',
                        delimiter="/")
    
    for file in keys:
        filepath = f"s3://{bucket_name}/{foldername}/{keys}"
        df_read = pd.read_csv(filepath, header=None)
    
    return df_read
    

# BUCKET_NAME = 'cap-stone-data-lake'

with DAG(
    dag_id='capstone-work-flow',
    default_args=default_args,
) as dag:
    s3_sensor_test = S3KeySensor(
        task_id='s3_sensor_test',
        bucket_key='s3://cap-stone-data-lake/test/', 
        bucket_name=BUCKET_NAME,
        aws_conn_id='aws_default',
        wildcard_match=True 
    )
    home_folder_get_test_data = S3FileTransformOperator(
        task_id="home_folder_get_test_data", 
        description='clean and transform home type building',
        source_s3_key='s3://cap-stone-data-lake/cleaned/*.csv',
        dest_s3_key='s3://cap-stone-data-lake/home/home_test.csv',
        replace=False,
        transform_script='/Users/rayno/airflow/dags/scripts/home_transform.py',
        source_aws_conn_id='aws_default',
        dest_aws_conn_id='aws_default'
    )
    office_folder_get_test_data = S3FileTransformOperator(
        task_id="office_folder_get_test_data", 
        description='clean and transform office type building',
        source_s3_key='s3://cap-stone-data-lake/cleaned/*.csv',
        dest_s3_key='s3://cap-stone-data-lake/office/office_test.csv',
        replace=False,
        transform_script='/Users/rayno/airflow/dags/scripts/office_transform.py',
        source_aws_conn_id='aws_default',
        dest_aws_conn_id='aws_default'
    )
    hotel_folder_get_test_data = S3FileTransformOperator(
        task_id="hotel_folder_get_test_data", 
        description='clean and transform hotel type building',
        source_s3_key='s3://cap-stone-data-lake/cleaned/*.csv',
        dest_s3_key='s3://cap-stone-data-lake/hotel/hotel_test.csv',
        replace=False,
        transform_script='/Users/rayno/airflow/dags/scripts/hotel_transform.py',
        source_aws_conn_id='aws_default',
        dest_aws_conn_id='aws_default'
    )
    read_from_train_bucket = PythonOperator(
        task_id='read_from_train_bucket',
        python_callable=read_train_data,
        provide_context=True
    )
    train_predictor = PythonOperator(
        task_id='train_predictor',
        python_callable=train_predictor,
        provide_context=True
    )
    s3_sensor_home = S3KeySensor(
        task_id='s3_sensor_home',
        bucket_key='s3://cap-stone-data-lake/home/', 
        bucket_name=BUCKET_NAME,
        aws_conn_id='aws_default',
        wildcard_match=True 
    )
    predict_home_bucket = PythonOperator(
        task_id='predict_home_bucket',
        python_callable=predict_home_test,
        provide_context=True
    )
    s3_sensor_office = S3KeySensor(
        task_id='s3_sensor_office',
        bucket_key='s3://cap-stone-data-lake/office/', 
        bucket_name=BUCKET_NAME,
        aws_conn_id='aws_default',
        wildcard_match=True 
    )
    predict_office_bucket = PythonOperator(
        task_id='predict_office_bucket',
        python_callable=predict_office_test,
        provide_context=True
    )
    s3_sensor_hotel = S3KeySensor(
        task_id='s3_sensor_hotel',
        bucket_key='s3://cap-stone-data-lake/hotel/', 
        bucket_name=BUCKET_NAME,
        aws_conn_id='aws_default',
        wildcard_match=True 
    )
    predict_hotel_bucket = PythonOperator(
        task_id='predict_hotel_bucket',
        python_callable=predict_hotel_test,
        provide_context=True
    )
    create_prediction_table = PostgresOperator(               
        task_id="create_prediction_table",                   
        postres_conn_id='postgres_default',
        sql="""
            CREATE TABLE IF NOT EXISTS Prediction_Result (
            Property ID SERIAL PRIMARY KEY,    
            Property Name VARCHAR NOT NULL,
            energy_predict VARCHAR NOT NULL,
            );
          """,
    )
    update_prediction_table = PythonOperator(
        task_id='update_prediction_table',
        python_callable=update_postgre_table,
        provide_context=True
    )
    
s3_sensor_test >> home_folder_get_test_data >> office_folder_get_test_data >> hotel_folder_get_test_data >> read_from_train_bucket >> train_predictor >> s3_sensor_home >> predict_home_bucket >> s3_sensor_office >> predict_office_test >> s3_sensor_hotel >> predict_hotel_test >> create_prediction_table >> update_prediction_table



