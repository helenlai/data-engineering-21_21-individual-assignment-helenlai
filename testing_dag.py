import datetime
import os

from airflow.models import Variable,DAG
from airflow.contrib.operators import dataproc_operator
from airflow.utils import trigger_rule
import logging


log = logging.getLogger(__name__)


yesterday = datetime.datetime.combine(
    datetime.datetime.today() - datetime.timedelta(1),
    datetime.datetime.min.time())

default_dag_args = {
    'start_date': yesterday,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
    'project_id': Variable.get('gcp_project')
}

with DAG(
        'creating_cluster',
        schedule_interval=datetime.timedelta(days=1),
        default_args=default_dag_args) as dag:
    
        log.info('creating cluster')
        create_dataproc_cluster = dataproc_operator.DataprocClusterCreateOperator(
            task_id='create_dataproc_cluster',
            # Give the cluster a unique name by appending the date scheduled.
            # See https://airflow.apache.org/code.html#default-variables
            cluster_name=Variable.get('cluster_name'),
            num_workers=2,
            image_version="1.4",
            init_actions_uris =["gs://goog-dataproc-initialization-actions-us-central1/python/pip-install.sh"],
            region=Variable.get('region'),
            metadata={"PIP_PACKAGES":"spark-nlp==2.7.5 fsspec gcsfs"},
            #zone=Variable.get('gce_zone'),
            master_machine_type='n1-standard-2',
            worker_machine_type='n1-standard-2')
        
            # Delete Cloud Dataproc cluster.
        delete_dataproc_cluster = dataproc_operator.DataprocClusterDeleteOperator(
            task_id='delete_dataproc_cluster',
            cluster_name=Variable.get('cluster_name'),
            region=Variable.get('region'),
            # Setting trigger_rule to ALL_DONE causes the cluster to be deleted
            # even if the Dataproc job fails.
            trigger_rule=trigger_rule.TriggerRule.ALL_DONE)
        
        create_dataproc_cluster >> delete_dataproc_cluster
