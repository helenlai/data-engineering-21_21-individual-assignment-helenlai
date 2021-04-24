import datetime
import os

from airflow.models import Variable,DAG
from airflow.contrib.operators import dataproc_operator
from airflow.operators.dummy_operator import DummyOperator
from airflow.utils import trigger_rule
import logging



yesterday = datetime.datetime.combine(
    datetime.datetime.today() - datetime.timedelta(1),
    datetime.datetime.min.time())

default_dag_args = {
    'start_date': yesterday,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': datetime.timedelta(minutes=5),
    'project_id': Variable.get('gcp_project')
}

with DAG(
        'testing',
        schedule_interval=datetime.timedelta(days=1),
        default_args=default_dag_args) as dag:

        submit_pyspark_job=dataproc_operator.DataProcPySparkOperator(
            task_id='submit_pyspark_job',
            main=Variable.get('spark_script_path'),
            job_name=Variable.get('job_name'),
            region=Variable.get('region'),
            cluster_name=Variable.get('cluster_name'),
            dataproc_pyspark_jars=Variable.get('jar_path')
        )

        start_task=DummyOperator(task_id='start_task',dag=dag)

        start_task>>submit_pyspark_job