

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
missing_values = ['unknown']
df = pd.read_csv('tcd ml 2019-20 income prediction training (with labels).csv', index_col = "Instance", na_values=missing_values)

```


```python
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

```

    Year of Record         441
    Gender               14281
    Age                    494
    Country                  0
    Size of City             0
    Profession             322
    University Degree     7370
    Wears Glasses            0
    Hair Color            7242
    Body Height [cm]         0
    Income in EUR            0
    dtype: int64
    (111993, 11)
    Year of Record        441
    Gender                  0
    Age                     0
    Country                 0
    Size of City            0
    Profession            322
    University Degree       0
    Wears Glasses           0
    Hair Color           7271
    Body Height [cm]        0
    Income in EUR           0
    dtype: int64
    (103858, 11)



```python
#Delete outliers with z score
mean = df['Income in EUR'].mean()
std = df['Income in EUR'].std()
print(df.shape)
df = df[(df['Income in EUR']-mean)/std<3]
print(df.shape)
df = df[(mean-df['Income in EUR'])/std<3]
print(df.shape)
```

    (103858, 11)
    (101648, 11)
    (101648, 11)



```python
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

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year of Record</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Country</th>
      <th>Size of City</th>
      <th>Profession</th>
      <th>University Degree</th>
      <th>Wears Glasses</th>
      <th>Hair Color</th>
      <th>Body Height [cm]</th>
      <th>Income</th>
    </tr>
    <tr>
      <th>Instance</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>111994</th>
      <td>1992.0</td>
      <td>other</td>
      <td>21.0</td>
      <td>Honduras</td>
      <td>391652</td>
      <td>senior project analyst</td>
      <td>2</td>
      <td>1</td>
      <td>Brown</td>
      <td>153</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>111995</th>
      <td>1986.0</td>
      <td>other</td>
      <td>34.0</td>
      <td>Kyrgyzstan</td>
      <td>33653</td>
      <td>greeter</td>
      <td>1</td>
      <td>0</td>
      <td>Black</td>
      <td>163</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>111996</th>
      <td>1994.0</td>
      <td>missing</td>
      <td>53.0</td>
      <td>Portugal</td>
      <td>34765</td>
      <td>liaison</td>
      <td>1</td>
      <td>1</td>
      <td>Blond</td>
      <td>153</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>111997</th>
      <td>1984.0</td>
      <td>missing</td>
      <td>29.0</td>
      <td>Uruguay</td>
      <td>1494132</td>
      <td>occupational therapist</td>
      <td>0</td>
      <td>0</td>
      <td>Black</td>
      <td>154</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>111998</th>
      <td>2007.0</td>
      <td>other</td>
      <td>17.0</td>
      <td>Serbia</td>
      <td>120661</td>
      <td>portfolio manager</td>
      <td>0</td>
      <td>0</td>
      <td>Red</td>
      <td>191</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
all_data = pd.concat((df,df_test))
for column in all_data.select_dtypes(include=[np.object]).columns:
    df[column] = df[column].astype('category', categories = all_data[column].unique())
    df_test[column] = df_test[column].astype('category', categories = all_data[column].unique())
```

    /usr/local/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:1: FutureWarning: Sorting because non-concatenation axis is not aligned. A future version
    of pandas will change to not sort by default.
    
    To accept the future behavior, pass 'sort=False'.
    
    To retain the current behavior and silence the warning, pass 'sort=True'.
    
      """Entry point for launching an IPython kernel.
    /usr/local/anaconda3/lib/python3.7/site-packages/IPython/core/interactiveshell.py:3325: FutureWarning: specifying 'categories' or 'ordered' in .astype() is deprecated; pass a CategoricalDtype instead
      exec(code_obj, self.user_global_ns, self.user_ns)



```python
#One-hot encoding
dummies_gender = pd.get_dummies(df.Gender)
dummies_country = pd.get_dummies(df.Country)
dummies_profession = pd.get_dummies(df.Profession)
dummies_haircolor = pd.get_dummies(df['Hair Color'])
merged = pd.concat([df,dummies_gender,dummies_country,dummies_profession,dummies_haircolor], axis = 'columns')
y = merged['Income in EUR']
merged = merged.drop(['Gender','Country','other', 'Income in EUR','Profession','Hair Color'],axis='columns')
merged.isnull().sum()
```




    Year of Record                     0
    Age                                0
    Size of City                       0
    University Degree                  0
    Wears Glasses                      0
    Body Height [cm]                   0
    missing                            0
    female                             0
    male                               0
    Belarus                            0
    Singapore                          0
    Norway                             0
    Cuba                               0
    United Arab Emirates               0
    Liberia                            0
    State of Palestine                 0
    Israel                             0
    South Sudan                        0
    Kyrgyzstan                         0
    Togo                               0
    Finland                            0
    Papua New Guinea                   0
    Paraguay                           0
    Belgium                            0
    Costa Rica                         0
    Senegal                            0
    Congo                              0
    Slovakia                           0
    Burundi                            0
    Portugal                           0
                                      ..
    animalbreeder                      0
    air & noise pollution inspector    0
    community coordinator              0
    accountable project manager        0
    collector                          0
    computer aide                      0
    blake fellow                       0
    accountant                         0
    audit supervisor                   0
    cartographer                       0
    asset management specialist        0
    administrative manager             0
    account executive                  0
    astronomer                         0
    brokerage clerk                    0
    asset manager                      0
    computer associate                 0
    administrative coordinator         0
    certified it administrator         0
    cashier                            0
    community assistant                0
    aerospace engineer                 0
    apparel patternmaker               0
    clinical case supervisor           0
    baggage porter                     0
    Blond                              0
    Black                              0
    Brown                              0
    Red                                0
    Unknown                            0
    Length: 1523, dtype: int64




```python
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

#This is to find the best regression model for the dataset; while it takes too much time run it fully, I commented it out from the inline result
'''
X_train, X_test, Y_train, Y_test = train_test_split (merged, y, test_size = 0.20, random_state=42)
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor


pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',LinearRegression())])))
pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO', Lasso())])))
pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN', ElasticNet())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])))
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
```




    '\nX_train, X_test, Y_train, Y_test = train_test_split (merged, y, test_size = 0.20, random_state=42)\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn.linear_model import Lasso\nfrom sklearn.linear_model import ElasticNet\nfrom sklearn.tree import DecisionTreeRegressor\nfrom sklearn.neighbors import KNeighborsRegressor\nfrom sklearn.ensemble import GradientBoostingRegressor\n\n\npipelines = []\npipelines.append((\'ScaledLR\', Pipeline([(\'Scaler\', StandardScaler()),(\'LR\',LinearRegression())])))\npipelines.append((\'ScaledLASSO\', Pipeline([(\'Scaler\', StandardScaler()),(\'LASSO\', Lasso())])))\npipelines.append((\'ScaledEN\', Pipeline([(\'Scaler\', StandardScaler()),(\'EN\', ElasticNet())])))\npipelines.append((\'ScaledKNN\', Pipeline([(\'Scaler\', StandardScaler()),(\'KNN\', KNeighborsRegressor())])))\npipelines.append((\'ScaledCART\', Pipeline([(\'Scaler\', StandardScaler()),(\'CART\', DecisionTreeRegressor())])))\npipelines.append((\'ScaledGBM\', Pipeline([(\'Scaler\', StandardScaler()),(\'GBM\', GradientBoostingRegressor())])))\n\nresults = []\nnames = []\nfor name, model in pipelines:\n    kfold = KFold(n_splits=10, random_state=21)\n    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=\'neg_mean_squared_error\')\n    results.append(cv_results)\n    names.append(name)\n    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())\n    print(msg)\n'




```python
from sklearn.linear_model import LassoCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
#Standardize the data
scaler = StandardScaler()
merged = scaler.fit_transform(merged)
model = LassoCV(normalize= False, max_iter = 100000)
model.fit(merged, y)

```

    /usr/local/anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:1978: FutureWarning: The default value of cv will change from 3 to 5 in version 0.22. Specify it explicitly to silence this warning.
      warnings.warn(CV_WARNING, FutureWarning)





    LassoCV(alphas=None, copy_X=True, cv='warn', eps=0.001, fit_intercept=True,
            max_iter=100000, n_alphas=100, n_jobs=None, normalize=False,
            positive=False, precompute='auto', random_state=None,
            selection='cyclic', tol=0.0001, verbose=False)




```python
model.score(merged,y)
```




    0.8684407028274769




```python
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
```


```python
new_merged = scaler.transform(new_merged)
result = model.predict(new_merged)
```


```python
df_result = pd.read_csv('tcd ml 2019-20 income prediction test (without labels).csv', na_values=missing_values,index_col='Instance')
```


```python
print(result)
df_result['Income'] = np.exp(result)
```

    [10.20945918  9.65002578 10.4170901  ... 11.1609789  12.06729418
     11.89249082]



```python
df_result[df_result['Income']<0]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year of Record</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Country</th>
      <th>Size of City</th>
      <th>Profession</th>
      <th>University Degree</th>
      <th>Wears Glasses</th>
      <th>Hair Color</th>
      <th>Body Height [cm]</th>
      <th>Income</th>
    </tr>
    <tr>
      <th>Instance</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
df_result = df_result.drop(df_result.columns[[0,1,2,3,4,5,6,7,8,9]],axis=1)
df_result.to_csv('Result4.0 Lasso with hair color and missing gender as new column.csv')
```


```python
df_result.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Income</th>
    </tr>
    <tr>
      <th>Instance</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>111994</th>
      <td>27158.875654</td>
    </tr>
    <tr>
      <th>111995</th>
      <td>15522.188283</td>
    </tr>
    <tr>
      <th>111996</th>
      <td>33426.026069</td>
    </tr>
    <tr>
      <th>111997</th>
      <td>137197.274315</td>
    </tr>
    <tr>
      <th>111998</th>
      <td>16035.246095</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
