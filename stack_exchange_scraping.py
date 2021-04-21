import json
import requests     # HTTP requests
# try:
#   from selenium import webdriver
# except:
#   !pip install selenium
#   from selenium import webdriver
# The following packages will also be used in this tutorial
import numpy as np  # Numerical operations
import time         # Tracking time
import re           # String manipulation
import pickle
import gcsfs
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
import datetime
from airflow import models



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
        if  response.json()['quota_remaining']==50:
            print('Almost no quota left')
            break


        print(len(response_item))

        for i in range(len(response_item)):
            tags=",".join(response_item[i]['tags'])
            question_body=response_item[i]['body']
            #question_body_cleaned=cleaning_response(question_body)
            question_body_cleaned=question_body
            question_title=response_item[i]['title']
            question_outcome=response_item[i]['is_answered']
            question_id=response_item[i]['question_id']
            
            question_id_lst.append(question_id)
            question_body_lst.append(question_body_cleaned)
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
    #tag=kwargs['tag']
    tag='machine-learning'
    print(tag)
    question_id_lst,question_body_lst,question_title_lst,question_outcome_list=scrape(tag,question_id_lst,question_body_lst,question_title_lst,question_outcome_list)
    data_lst=clean(question_id_lst,question_body_lst,question_title_lst,question_outcome_list)
    full_data_name_lst=['question_id_lst','question_body_lst','question_title_lst','question_outcome_list']
    
    fs = gcsfs.GCSFileSystem(project=kwargs['project_id'])
    for data,name in zip(data_lst,full_data_name_lst):
        with fs.open(kwargs['bucket_name'] + '/data/' + name, 'wb') as handle:
            pickle.dump(data,handle)
    
    
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
    'project_id': models.Variable.get('gcp_project'),
    'bucket_name':models.Variable.get('gcs_bucket'),
    'tag':models.Variable.get('data_name')
}

with models.DAG(
        'scrap_save_data',
        schedule_interval=datetime.timedelta(days=1),
        default_args=default_dag_args) as dag:
    
    scrap_save_data = PythonOperator(task_id='get_save_data',
                            python_callable=scrap_clean_save_data,
                            op_kwargs=default_dag_args,
                            provide_context=True)
    
    start_task=DummyOperator(task_id='start_task',dag=dag)
    
    start_task>>scrap_save_data
