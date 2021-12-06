import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
import csv
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import *
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_selector as selector
from sklearn.linear_model import SGDClassifier,SGDRegressor
from sklearn import metrics
import time
import re
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql.functions import udf,variance
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
import json
from pyspark.sql import Row
from pyspark.sql.types import StructType,StructField, StringType,FloatType
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.functions import isnan, when, count, col
from pyspark.sql import functions as F
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from pyspark.ml.evaluation import BinaryClassificationEvaluator
format_train=StructType([
    StructField('Dates', StringType(), True),
    StructField('Category', StringType(), True),
    StructField('Descript', StringType(), True),
    StructField('DayOfWeek', StringType(), True),
    StructField('PdDistrict', StringType(), True),
    StructField('Resolution', StringType(), True),
    StructField('Address', StringType(), True),
    StructField('X', FloatType(), True),
    StructField('Y', FloatType(), True)
])
format_test=StructType([
    StructField('Id', StringType(), True),
    StructField('Dates', StringType(), True),
    StructField('DayOfWeek', StringType(), True),
    StructField('PdDistrict', StringType(), True),
    StructField('Address', StringType(), True),
    StructField('X', FloatType(), True),
    StructField('Y', FloatType(), True)
])
sc = SparkContext.getOrCreate()
sc.setLogLevel("OFF")
ssc = StreamingContext(sc, 1)
spark=SparkSession(sc)
data = ssc.socketTextStream("localhost", 6100) 
df=pd.read_csv('C:/Users/nidhi/Downloads/train(2).csv',nrows=3000)
c_len=[i for i in range(40)]
classes=['ARSON','ASSAULT','BAD CHECKS','BRIBERY','BURGLARY','DISORDERLY CONDUCT','DRIVING UNDER THE INFLUENCE','DRUG/NARCOTIC','DRUNKENNESS','EMBEZZLEMENT','EXTORTION','FAMILY OFFENSES','FORGERY/COUNTERFEITING',\
         'FRAUD','GAMBLING','KIDNAPPING','LARCENY/THEFT','LIQUOR LAWS','LOITERING','MISSING PERSON','NON-CRIMINAL','OTHER OFFENSES','PORNOGRAPHY','PORNOGRAPHY/OBSCENE MAT','PROSTITUTION','RECOVERED VEHICLE','ROBBERY',\
         'RUNAWAY','SECONDARY CODES','SEX OFFENSES FORCIBLE','SEX OFFENSEES NON FORCIBLE','STOLEN PROPERTY','SUICIDE','SUSPICIOUS OCC','TREA','TRESPASS','VANDALISM','VEHICLE THEFT','WARRANTS','WEAPON LAWS']
days_list=['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']
address_list=list(np.unique(df['Address']))
pdDist_list=list(np.unique(df['PdDistrict']))
def transform(df,l):
    z=[]
    for i in df:
        if(i not in l):
            l.append(i)
        g=l.index(i)
        z.append(g)
    return z
def preprocess(df):

	hour=[]
	for time in df['Dates']:
    	x=time.split()
    	z=x[1].split(":")
    	hours=int(z[0])
    	mint=int(z[1])
    	if(mint>=30) and(hours<23):
        	hours+=1
    	elif (mint>=30) and(hours==23):
        	hours=0
    	hour.append(hours)
	df['Time']=hour
	#df=df.drop(['Dates','Descript','Resolution'],axis=1)
	df['Category']=transform(df['Category'],classes)
	df['DayOfWeek']=transform(df['DayOfWeek'],days_list)

	df['Address']=transform(df['Address'],address_list)
	df['PdDistrict']=transform(df['PdDistrict'],pdDist_list)
	return df
def readTrain(rdd): 
		
		if(len(rdd.collect()) > 0):
		  val = json.loads(rdd.collect()[0])
		  keys = val.keys()
		  df = spark.createDataFrame([], format_train)
		  for key in val :
		    
		  
		    rdd1 = sc.parallelize([val[key]])
		    if(not rdd1.isEmpty()):
		      row = spark.read.json(rdd1)
		      if(len(row.columns) == 9):
		        df = df.union(row)
		  df = preprocessing(df)
		  df.show()
		  y=df['Category']
		  X=df.drop(columns=['Category'])
		  sgd_clf.partial_fit(x,y,classes=c_len)
		  clf.partial_fit(x,y,classes=c_len)
		  kmeans.partial_fit(x)
		  
def readTest(rdd): 
		
		if(len(rdd.collect()) > 0):
		  val = json.loads(rdd.collect()[0])
		  keys = val.keys()
		  df = spark.createDataFrame([], format_test)
		  for key in val :
		    
		  
		    rdd1 = sc.parallelize([val[key]])
		    if(not rdd1.isEmpty()):
		      row = spark.read.json(rdd1)
		      if(len(row.columns) == 7):
		        df = df.union(row)
		  df=df.drop(columns=['Id'])
		  df = preprocessing(df)
		  df.show()
		  sgd_pred=sgd_clf.predict(df)
		  sdgr_pred=clf.predict(df)
		  kmeans_pred=kmeans.predict(df)
		  print("SGDClassifier",sgd_pred,'\n','SGDRegressor',sgdr_pred,'\n','KMeans',kmeans_pred)
		  
		  

sgd_clf=SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
kmeans=MiniBarchKMeans(n_cluster=40,random_state=0,batch_size=100
clf=SGDRegressor(random_state=42, max_iter=1000, tol=1e-3)
data.foreachRDD(lambda rdd: readTrain(rdd))
data.foreachRDD(lambda rdd: readTest(rdd))
ssc.start()
ssc.awaitTermination()





