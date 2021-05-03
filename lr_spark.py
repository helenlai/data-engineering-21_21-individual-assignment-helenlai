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
input_df.show(4)



document_assembler = DocumentAssembler() \
    .setInputCol("question") \
    .setOutputCol("document")
# converting question content into arrays of tokens
tokenizer = Tokenizer() \
  .setInputCols(["document"]) \
  .setOutputCol("token")
normalizer = Normalizer() \
    .setInputCols(["token"]) \
    .setOutputCol("normalized")
# remove stopwords based on the dafulat stopword dictionary
stopwords_cleaner = StopWordsCleaner()\
      .setInputCols("normalized")\
      .setOutputCol("cleanTokens")\
      .setCaseSensitive(False)
# converting tokens to its root form 
stemmer = Stemmer() \
    .setInputCols(["cleanTokens"]) \
    .setOutputCol("stem")
#output document structure as token arrays

finisher = Finisher() \
    .setInputCols(["stem"]) \
    .setOutputCols(["token_features"]) \
    .setOutputAsArray(True) \
    .setCleanAnnotations(False)
# Compute TF-IDF as text featurisation
hashingTF = HashingTF(inputCol="token_features", outputCol="rawFeatures", numFeatures=1000)# To generate Inverse Document Frequency
idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=5)# convert labels (string) to integers. Easy to process compared to string.
#setting the regularisation parameter
REG=0.3
# Initialisation the logistic regression
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

#inference on train,test and validation set
pipeline_model=nlp_pipeline.fit(input_df)
tr_pred=pipeline_model.transform(input_df)
val_pred=pipeline_model.transform(val_df)
test_pred=pipeline_model.transform(test_df)

#saving the trained pipeline model
pipeline_model.save(prefix+"/models/lr")


evaluator = MulticlassClassificationEvaluator(
    labelCol="outcome", predictionCol="prediction", metricName="accuracy")

#computing accuracy on train, test and validation set
evaluator.evaluate(val_pred)
acc_lst=[]
for pred in [tr_pred,val_pred,test_pred]:    
    accuracy = evaluator.evaluate(pred)
    acc_lst.append(accuracy)
    print("Accuracy = %g" % (accuracy))
   
#saving result
fs = gcsfs.GCSFileSystem(project=PROJECT_ID)
with fs.open(prefix+'/results/result_lr.txt','w') as handle:
    handle.write(str(acc_lst))
    











