#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
missing_values = ['unknown']
df = pd.read_csv('tcd ml 2019-20 income prediction training (with labels).csv', index_col = "Instance", na_values=missing_values)

print(df.isnull().sum())
print(df.shape)

#Convert text features to ordinal
df['University Degree'].replace(to_replace = "0", value ='No', inplace= True)
df['University Degree'].replace(to_replace = np.nan, value ='No', inplace= True)
df['University Degree'].replace(to_replace = "No", value = 0, inplace= True)
df['University Degree'].replace(to_replace = "Bachelor", value = 1, inplace= True)
df['University Degree'].replace(to_replace = "Master", value = 2, inplace= True)
df['University Degree'].replace(to_replace = "PhD", value = 4, inplace= True)

#Delete and fill nan rows
df['Gender'].replace(to_replace = "0", value = np.nan, inplace= True)
df['Gender'].fillna('missing', inplace= True)
df['Hair Color'].replace(to_replace = "0", value = np.nan, inplace= True)
#df['Year of Record'].fillna(df['Year of Record'].median(), inplace=True)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Body Height [cm]'].fillna(df['Body Height [cm]'].median(), inplace = True)
print(df.isnull().sum())

df = df.dropna(subset=['Gender','Age','Year of Record','Country','Size of City','Profession','Wears Glasses','Body Height [cm]','Hair Color'])

#Delete garbage data
df = df[df['Income in EUR']>0]
df.isnull().sum()
print(df.shape)


#Delete outliers with z score
mean = df['Income in EUR'].mean()
std = df['Income in EUR'].std()
print(df.shape)
df = df[(df['Income in EUR']-mean)/std<3]
print(df.shape)
df = df[(mean-df['Income in EUR'])/std<3]
print(df.shape)

df_test = pd.read_csv('tcd ml 2019-20 income prediction test (without labels).csv', index_col = "Instance", na_values=missing_values)

#Convert text feature to ordinal
df_test['University Degree'].replace(to_replace = "0", value ='No', inplace= True)
df_test['University Degree'].replace(to_replace = np.nan, value ='No', inplace= True)
df_test['University Degree'].replace(to_replace = "No", value = 0, inplace= True)
df_test['University Degree'].replace(to_replace = "Bachelor", value = 1, inplace= True)
df_test['University Degree'].replace(to_replace = "Master", value = 2, inplace= True)
df_test['University Degree'].replace(to_replace = "PhD", value = 4, inplace= True)

#Fill nan rows
df_test['Year of Record'].fillna(df_test['Year of Record'].median(), inplace=True)
df_test['Gender'].replace(to_replace = "0", value = np.nan, inplace= True)
df_test['Gender'].fillna('missing', inplace=True)
df_test['Age'].fillna(df_test['Age'].median(), inplace=True)
df_test['Body Height [cm]'].fillna(df_test['Body Height [cm]'].median(), inplace = True)
df_test['Hair Color'].replace(to_replace = "0", value = np.nan, inplace= True)

df_test['Country'].fillna(df_test['Country'].value_counts().idxmax(), inplace=True)
df_test['Size of City'].fillna(df_test['Size of City'].mean(), inplace=True)
df_test['Profession'].fillna(df_test['Profession'].value_counts().idxmax(), inplace=True)
df_test['Hair Color'].fillna(df_test['Hair Color'].value_counts().idxmax(), inplace=True)
df_test['Wears Glasses'].fillna(df_test['Wears Glasses'].value_counts().idxmax(), inplace=True)

df_test.head()


all_data = pd.concat((df,df_test))
for column in all_data.select_dtypes(include=[np.object]).columns:
    df[column] = df[column].astype('category', categories = all_data[column].unique())
    df_test[column] = df_test[column].astype('category', categories = all_data[column].unique())

#One-hot encoding
dummies_gender = pd.get_dummies(df.Gender)
dummies_country = pd.get_dummies(df.Country)
dummies_profession = pd.get_dummies(df.Profession)
dummies_haircolor = pd.get_dummies(df['Hair Color'])
merged = pd.concat([df,dummies_gender,dummies_country,dummies_profession,dummies_haircolor], axis = 'columns')
y = merged['Income in EUR']
merged = merged.drop(['Gender','Country','other', 'Income in EUR','Profession','Hair Color'],axis='columns')
merged.isnull().sum()

#log the big number data
merged['Size of City'] = np.log(merged['Size of City'])
merged['Year of Record'] = np.log(merged['Year of Record'])
merged['Body Height [cm]'] = np.log(merged['Body Height [cm]'])
y = np.log(y)
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#This is to find the best regression model for the dataset, but it takes too long to run everytime
#So I commented it out
'''
X_train, X_test, Y_train, Y_test = train_test_split (merged, y, test_size = 0.20, random_state=42)
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor


pipelines = []
#pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',LinearRegression())])))
#pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO', Lasso())])))
#pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN', ElasticNet())])))
#pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeRegressor())])))
pipelines.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingRegressor())])))

results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=10, random_state=21)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='neg_mean_squared_error')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

'''



from sklearn.linear_model import LassoCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
#Standardize the data
scaler = StandardScaler()
merged = scaler.fit_transform(merged)
model = LassoCV(normalize= False, max_iter = 100000)
model.fit(merged, y)
model.score(merged,y)

#One-hot encoding
dummies_gender = pd.get_dummies(df_test.Gender)
dummies_country = pd.get_dummies(df_test.Country)
dummies_profession = pd.get_dummies(df_test.Profession)
dummies_haircolor = pd.get_dummies(df_test['Hair Color'])
new_merged = pd.concat([df_test,dummies_gender,dummies_country,dummies_profession,dummies_haircolor], axis = 'columns')
#merged = merged.drop(['Gender','Country','Profession','other', 'Income'],axis='columns')
new_merged = new_merged.drop(['Gender','Country','other', 'Income','Profession','Hair Color'],axis='columns')

new_merged['Size of City'] = np.log(new_merged['Size of City'])
new_merged['Year of Record'] = np.log(new_merged['Year of Record'])
new_merged['Body Height [cm]'] = np.log(new_merged['Body Height [cm]'])

new_merged = scaler.transform(new_merged)
result = model.predict(new_merged)

#Reread the file without label, delete irrevalant rows other than Income and Instances
df_result = pd.read_csv('tcd ml 2019-20 income prediction test (without labels).csv', na_values=missing_values,index_col='Instance')

print(result)
#Transform the logged Income back
df_result['Income'] = np.exp(result)
df_result[df_result['Income']<0]
df_result = df_result.drop(df_result.columns[[0,1,2,3,4,5,6,7,8,9]],axis=1)
df_result.to_csv('Result4.0 Lasso with hair color and missing gender as new column.csv')

df_result.head()
