import pandas as pd
import pickle
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.feature import HashingTF, IDF, StringIndexer, SQLTransformer,IndexToString

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.classification import LogisticRegression


from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp.pretrained import PretrainedPipeline
import sparknlp
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
from pyspark.sql import SQLContext
from pyspark.ml.evaluation import BinaryClassificationEvaluator 

import gcsfs


spark = SparkSession.builder \
    .appName("Stack Exchange Text Classification")\
    .getOrCreate()



tr_df=pd.read_pickle('gs://europe-west2-de-composer-en-47f87717-bucket/data/tr_df.pkl')
val_df=pd.read_pickle('gs://europe-west2-de-composer-en-47f87717-bucket/data/val_df.pkl')
test_df=pd.read_pickle('gs://europe-west2-de-composer-en-47f87717-bucket/data/test_df.pkl')



sqlContext = SQLContext(spark)
input_df=sqlContext.createDataFrame(tr_df)
val_df=sqlContext.createDataFrame(val_df)
test_df=sqlContext.createDataFrame(test_df)
input_df.show(4)


REG=0.3
document_assembler = DocumentAssembler() \
    .setInputCol("question") \
    .setOutputCol("document")# convert document to array of tokens
tokenizer = Tokenizer() \
  .setInputCols(["document"]) \
  .setOutputCol("token")
normalizer = Normalizer() \
    .setInputCols(["token"]) \
    .setOutputCol("normalized")# remove stopwords
stopwords_cleaner = StopWordsCleaner()\
      .setInputCols("normalized")\
      .setOutputCol("cleanTokens")\
      .setCaseSensitive(False)# stems tokens to bring it to root form
stemmer = Stemmer() \
    .setInputCols(["cleanTokens"]) \
    .setOutputCol("stem")# Convert custom document structure to array of tokens.


finisher = Finisher() \
    .setInputCols(["stem"]) \
    .setOutputCols(["token_features"]) \
    .setOutputAsArray(True) \
    .setCleanAnnotations(False)# To generate Term Frequency

hashingTF = HashingTF(inputCol="token_features", outputCol="rawFeatures", numFeatures=1000)# To generate Inverse Document Frequency
idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=5)# convert labels (string) to integers. Easy to process compared to string.

lr = LogisticRegression(featuresCol="features", labelCol='outcome', regParam=REG)

nlp_pipeline = Pipeline(
    stages=[document_assembler, 
            tokenizer,
            normalizer,
            stopwords_cleaner, 
            stemmer, 
            finisher,
            hashingTF,
            idf,
            lr])


pipeline_model=nlp_pipeline.fit(input_df)
tr_pred=pipeline_model.transform(input_df)
val_pred=pipeline_model.transform(val_df)
test_pred=pipeline_model.transform(test_df)


from pyspark.ml.evaluation import MulticlassClassificationEvaluator


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

with fs.open('gs://europe-west2-de-composer-en-47f87717-bucket/result/result_lr.txt','wb') as handle:
    pickle.dump(acc_lst,handle)
    



classsifierdl = ClassifierDLApproach()\
.setInputCols(["sentence_embeddings"])\
.setOutputCol("class")\
.setLabelColumn("outcome")\
.setMaxEpochs(5)\
.setEnableOutputLogs(True)













