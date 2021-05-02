
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.models import Variable,DAG
from airflow.contrib.operators import dataproc_operator
from airflow.utils import trigger_rule
import datetime
from airflow.models import Variable,DAG
import gcsfs


import datetime
import os
import logging
import json
import requests     # HTTP requests
# The following packages will also be used in this tutorial
import numpy as np  # Numerical operations
import pandas as pd
import time         # Tracking time
import re           # String manipulation
import pickle


def get_replacements(html_tags_open):
    html_tags_close=[]
    for tag in html_tags_open:
        html_tags_close.append(tag[0:1]+'/'+tag[1::])
    replacements = [
    ('<code>.*?</code>',' code shown here '),
    ('http\S+',''),
    ('<a href.*?>',''),
    ('<img src.*?>', ''),
    ('<pre class.*?>', ''),
    ('\n', ''),
    ('$',''),
    ('-+','')]
    for tag_open,tag_close in zip(html_tags_open,html_tags_close):
        replacements.append((tag_open,''))
        replacements.append((tag_close,''))
    return replacements
    


def cleaning_response(item):
    
    html_tags_open=['<a>','<b>','<p>','<pre>','<blockquote>','<del>','<dd>','<dl>','<dt>','<em>','<h1>','<h2>','<h3>','<i>','<img>','<kbd>','<li>','<ol>','<s>','<sup>','<sub>','<strong>','<ul>','<br>','<hr>']
    replacements=get_replacements(html_tags_open)
    
    for patter_to_replace, replacement in replacements:
        item = re.sub(patter_to_replace, replacement, item,flags=re.DOTALL)
    item=item.replace('$','').replace('#','').replace('=','')
    return item


def scrape(tag,question_id_lst,question_body_lst,question_title_lst,question_outcome_list):
    
    page_num=1
    while True:
        url='https://api.stackexchange.com/2.2/search/advanced?page='+str(page_num)+'&pagesize=100&todate=1585699200&order=desc&sort=activity&tagged='+tag+'&site=stackoverflow&filter=!--1nZw8Pr5S*'
        response=requests.get(url)
        print(response)
        print(response.json())
        response_item=response.json()['items']
        print(response_item)

        if response.json()['has_more']:
            page_num+=1
        else:
            print('no more questions to be scraped')
            break
        if response.json()['quota_remaining']<=50:
            print('Limited quota left!')
        if  response.json()['quota_remaining']==0:
            print('No quota left')
            break


        print(len(response_item))

        for i in range(len(response_item)):
            tags=",".join(response_item[i]['tags'])
            question_body=response_item[i]['body']
            #question_body_cleaned=cleaning_response(question_body)
            #question_body_cleaned=question_body
            question_title=response_item[i]['title']
            question_outcome=response_item[i]['is_answered']
            question_id=response_item[i]['question_id']
            
            question_id_lst.append(question_id)
            question_body_lst.append(question_body)
            question_title_lst.append(question_title)
            question_outcome_list.append(question_outcome)
        print(len(question_body_lst))
        
        if page_num==4:
            break
    
            
    return question_id_lst,question_body_lst,question_title_lst,question_outcome_list


def clean(question_id_lst,question_body_lst,question_title_lst,question_outcome_list):
    question_body_cleaned_lst=[]
    for question in question_body_lst:
        question_body_cleaned_lst.append(cleaning_response(question))
        
    question_body_cleaned_lst_final=[]
    idx_to_drop=[]
    for i,q in enumerate(question_body_cleaned_lst):
        if 'Error' in q:
            idx_to_drop.append(i)
        else:
            question_body_cleaned_lst_final.append(q)
            
    full_data_lst=[question_body_cleaned_lst_final]
    for data_lst in [question_id_lst,question_title_lst,question_outcome_list]:
        full_data_lst.append(np.delete(data_lst,idx_to_drop))
        
    len_lst=[len(full_data_lst[i]) for i in range(len(full_data_lst))]
    assert len_lst.count(len_lst[0])==len(len_lst)

    return full_data_lst
    

def scrap_clean_save_data(**kwargs):
    question_body_lst=[]
    question_title_lst=[]
    question_outcome_list=[]
    question_id_lst=[]
    tag=kwargs['tag']
    question_id_lst,question_body_lst,question_title_lst,question_outcome_list=scrape(tag,question_id_lst,question_body_lst,question_title_lst,question_outcome_list)
    question_body_lst,question_id_lst,question_title_lst,question_outcome_list=clean(question_id_lst,question_body_lst,question_title_lst,question_outcome_list)


    question_lst=[title+question for title,question in zip(question_title_lst,question_body_lst)]
    col_names=['question_id','question','outcome']
    data_df=pd.DataFrame(np.array([question_id_lst,question_lst,question_outcome_list]).T,columns=col_names)
    data_df['outcome']=(data_df.outcome=='True')*1
    data_idx=np.arange(len(data_df))
    np.random.shuffle(data_idx)
    num_test=round(float(kwargs['test_ratio'])*len(data_idx))
    num_val=round(float(kwargs['val_ratio'])*len(data_idx))
    test_df=data_df.iloc[data_idx[:num_test]]
    val_df=data_df.iloc[data_idx[num_test:num_test+num_val]]
    tr_df=data_df.iloc[data_idx[num_test+num_val::]]
    
    data_name_lst=["test_df","val_df","tr_df"]
    data_lst=[test_df,val_df,tr_df]
    fs = gcsfs.GCSFileSystem(project=kwargs['project_id'])
    for data,name in zip(data_lst,data_name_lst):
        data.to_parquet("gs://"+kwargs['bucket_name'] + '/data/' + name+'.parquet.gzip')
    
    
    return print('done scrapping, cleaning and saving')
    

yesterday = datetime.datetime.combine(
    datetime.datetime.today() - datetime.timedelta(1),
    datetime.datetime.min.time())
    
default_dag_args = {
    'start_date': yesterday,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': datetime.timedelta(minutes=5),
    'project_id': Variable.get('gcp_project'),
    'bucket_name':Variable.get('gcp_bucket'),
    'tag':Variable.get('tag'),
    'test_ratio':Variable.get('test_ratio'),
    'val_ratio':Variable.get('val_ratio'),
}


    
with DAG(
        'full_run_v2',
        schedule_interval=datetime.timedelta(days=1),
        default_args=default_dag_args) as dag:
    


        scrap_save_data = PythonOperator(task_id='get_save_data',
                        python_callable=scrap_clean_save_data,
                        op_kwargs=default_dag_args,
                        provide_context=True)

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
        
        submit_pyspark_job=dataproc_operator.DataProcPySparkOperator(
            task_id='submit_pyspark_job',
            main=Variable.get('spark_script_path'),
            job_name=Variable.get('job_name'),
            region=Variable.get('region'),
            cluster_name=Variable.get('cluster_name'),
            dataproc_pyspark_jars=Variable.get('jar_path'))

        delete_dataproc_cluster = dataproc_operator.DataprocClusterDeleteOperator(
            task_id='delete_dataproc_cluster',
            cluster_name=Variable.get('cluster_name'),
            region=Variable.get('region'),
            # Setting trigger_rule to ALL_DONE causes the cluster to be deleted
            # even if the Dataproc job fails.
            trigger_rule=trigger_rule.TriggerRule.ALL_DONE)

            # Delete Cloud Dataproc cluster.
     
        
        scrap_save_data>>create_dataproc_cluster>>submit_pyspark_job>>delete_dataproc_cluster
        
