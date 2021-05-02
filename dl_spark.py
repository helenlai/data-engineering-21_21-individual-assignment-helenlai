import pandas as pd
import pickle
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp.pretrained import PretrainedPipeline
import sparknlp
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
from pyspark.sql import SQLContext
from pyspark.ml.evaluation import BinaryClassificationEvaluator 
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import gcsfs


spark = SparkSession.builder \
    .appName("Stack Exchange Text Classification")\
    .getOrCreate()


prefix='gs://de-indv-bucket'

tr_df=pd.read_parquet(prefix+'/data/tr_df.parquet.gzip')
val_df=pd.read_parquet(prefix+'/data/val_df.parquet.gzip')
test_df=pd.read_parquet(prefix+'/data/test_df.parquet.gzip')



sqlContext = SQLContext(spark)
input_df=sqlContext.createDataFrame(tr_df)
val_df=sqlContext.createDataFrame(val_df)
test_df=sqlContext.createDataFrame(test_df)


document = DocumentAssembler()\
    .setInputCol('question')\
    .setOutputCol("out")

use = UniversalSentenceEncoder.pretrained()\
 .setInputCols(["out"])\
 .setOutputCol("sentence_embeddings")


classsifierdl = ClassifierDLApproach()\
  .setInputCols(["sentence_embeddings"])\
  .setOutputCol("class")\
  .setLabelColumn("outcome")\
  .setMaxEpochs(150)\
  .setBatchSize(32)\
  .setLr(0.005)\
  .setEnableOutputLogs(True)


use_clf_pipeline = Pipeline(
    stages = [
        document,
        use,
        classsifierdl
    ])



pipeline_model=use_clf_pipeline.fit(input_df)
tr_pred=pipeline_model.transform(input_df)
val_pred=pipeline_model.transform(val_df)
test_pred=pipeline_model.transform(test_df)


pipeline_model.save(prefix+"/models/dl")


evaluator = MulticlassClassificationEvaluator(
    labelCol="outcome", predictionCol="prediction", metricName="accuracy")


evaluator.evaluate(val_pred)
acc_lst=[]
for pred in [tr_pred,val_pred,test_pred]:    
    accuracy = evaluator.evaluate(pred)
    acc_lst.append(accuracy)
    print("Accuracy = %g" % (accuracy))

PROJECT_ID='de-indv-project'    
fs = gcsfs.GCSFileSystem(project=PROJECT_ID)

with fs.open(prefix+'/results/result_lr.txt','w') as handle:
    handle.write(str(acc_lst))
    













