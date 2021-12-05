import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
import skimage
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
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
i=0
"""with open('C:/Users/nidhi/Downloads/train(2).csv','r')as file:
    my_reader=csv.reader(file,delimiter=',')
    for row in my_reader:
        if(i>1500):
            break
        print(i,row)
        i+=1

"""
df=pd.read_csv('C:/Users/nidhi/Downloads/train(2).csv',nrows=3000)
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
print(df.head(10))
df=df.drop(['Dates','Descript','Resolution'],axis=1)
print(df.head(10))
numerical_columns_selector=selector(dtype_exclude=object)
categorical_columns_selector=selector(dtype_include=object)
categorical_preprocessor=OneHotEncoder(handle_unknown="ignore")
numerical_preprocessor=StandardScaler()
y=df['Category']
X=df.drop(columns=['Category'])
numerical_columns=numerical_columns_selector(X)
categorical_columns=categorical_columns_selector(X)
preprocessor=ColumnTransformer([
    ('one-hot-encoder',categorical_preprocessor,categorical_columns),
    ('standard-scaler',numerical_preprocessor,numerical_columns)])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
pipeline=make_pipeline(preprocessor,LogisticRegression(max_iter=500))
m=pipeline.fit(X,y)
df1=pd.read_csv('C:/Users/nidhi/Downloads/test(2).csv',nrows=3000)
hour=[]
for time in df1['Dates']:
    x=time.split()
    z=x[1].split(":")
    hours=int(z[0])
    mint=int(z[1])
    if(mint>=30) and(hours<23):
        hours+=1
    elif (mint>=30) and(hours==23):
        hours=0
    hour.append(hours)
df1['Time']=hour
print(df1.head(10))
df1=df1.drop(['Dates','Id'],axis=1)
y_pred=m.predict(df1)
print(y_pred)
#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("V",X)
p=pd.DataFrame(preprocessor.fit_transform(X))
print(p.head())
pipeline1=make_pipeline(preprocessor,SGDClassifier(random_state=42, max_iter=1000, tol=1e-3))
pipeline1.partial_fit(x_train,y_train)
pipeline1.partial_fit(x_test,y_test)
y_pred=pipeline1.predict(df1)
print(y_pred)




