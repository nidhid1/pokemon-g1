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
    
y_pred=m.predict(df1)
print(y_pred)
classes=['ARSON','ASSAULT','BAD CHECKS','BRIBERY','BURGLARY','DISORDERLY CONDUCT','DRIVING UNDER THE INFLUENCE','DRUG/NARCOTIC','DRUNKENNESS','EMBEZZLEMENT','EXTORTION','FAMILY OFFENSES','FORGERY/COUNTERFEITING',\
         'FRAUD','GAMBLING','KIDNAPPING','LARCENY/THEFT','LIQUOR LAWS','LOITERING','MISSING PERSON','NON-CRIMINAL','OTHER OFFENSES','PORNOGRAPHY','PORNOGRAPHY/OBSCENE MAT','PROSTITUTION','RECOVERED VEHICLE','ROBBERY',\
         'RUNAWAY','SECONDARY CODES','SEX OFFENSES FORCIBLE','SEX OFFENSEES NON FORCIBLE','STOLEN PROPERTY','SUICIDE','SUSPICIOUS OCC','TREA','TRESPASS','VANDALISM','VEHICLE THEFT','WARRANTS','WEAPON LAWS']
c_len=[i for i in range(40)]
category_list=np.unique(classes)
#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#data=categorical_preprocessor.fit_transform(df['Category'])
df['Category']=transform(df['Category'],classes)
df['DayOfWeek']=transform(df['DayOfWeek'],days_list)
df['Address']=transform(df['Address'],address_list)
df['PdDistrict']=transform(df['PdDistrict'],pdDist_list)
y=df['Category']
print(df['DayOfWeek'].head(10))
X=df.drop(columns=['Category'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
for i in y_test:
    if(i not in range(len(classes))):
        print(i)
sgd_clf=SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
sgd_clf.partial_fit(X_train,y_train,classes=c_len)
sgd_clf.partial_fit(X_test,y_test)
#df1['Category']=transform(df1['Category'],classes)
df1['DayOfWeek']=transform(df1['DayOfWeek'],days_list)
df1['Address']=transform(df1['Address'],address_list)
df1['PdDistrict']=transform(df1['PdDistrict'],pdDist_list)
y_pred=sgd_clf.predict(df1)
print(y_pred)




