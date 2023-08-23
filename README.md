# House Price Prediction


```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import umap
from sklearn.datasets import load_digits
from sklearn.decomposition import FastICA
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
```


```python
# load data
data = pd.read_csv('DataMining_HWI_Dataset.csv')

print(type(data))
data.head()
```

    <class 'pandas.core.frame.DataFrame'>


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rental_id</th>
      <th>building_id</th>
      <th>rent</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>size_sqft</th>
      <th>min_to_subway</th>
      <th>floor</th>
      <th>building_age_yrs</th>
      <th>no_fee</th>
      <th>has_roofdeck</th>
      <th>has_washer_dryer</th>
      <th>has_doorman</th>
      <th>has_elevator</th>
      <th>has_dishwasher</th>
      <th>has_patio</th>
      <th>has_gym</th>
      <th>neighborhood</th>
      <th>submarket</th>
      <th>borough</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1545</td>
      <td>44518357</td>
      <td>2550</td>
      <td>0.0</td>
      <td>1</td>
      <td>480</td>
      <td>9</td>
      <td>2.0</td>
      <td>17</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>Upper East Side</td>
      <td>All Upper East Side</td>
      <td>Manhattan</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2472</td>
      <td>94441623</td>
      <td>11500</td>
      <td>2.0</td>
      <td>2</td>
      <td>2000</td>
      <td>4</td>
      <td>1.0</td>
      <td>96</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Greenwich Village</td>
      <td>All Downtown</td>
      <td>Manhattan</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10234</td>
      <td>87632265</td>
      <td>3000</td>
      <td>3.0</td>
      <td>1</td>
      <td>1000</td>
      <td>4</td>
      <td>1.0</td>
      <td>106</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Astoria</td>
      <td>Northwest Queens</td>
      <td>Queens</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2919</td>
      <td>76909719</td>
      <td>4500</td>
      <td>1.0</td>
      <td>1</td>
      <td>916</td>
      <td>2</td>
      <td>51.0</td>
      <td>29</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Midtown</td>
      <td>All Midtown</td>
      <td>Manhattan</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2790</td>
      <td>92953520</td>
      <td>4795</td>
      <td>1.0</td>
      <td>1</td>
      <td>975</td>
      <td>3</td>
      <td>8.0</td>
      <td>31</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>Greenwich Village</td>
      <td>All Downtown</td>
      <td>Manhattan</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.describe()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rental_id</th>
      <th>building_id</th>
      <th>rent</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>size_sqft</th>
      <th>min_to_subway</th>
      <th>floor</th>
      <th>building_age_yrs</th>
      <th>no_fee</th>
      <th>has_roofdeck</th>
      <th>has_washer_dryer</th>
      <th>has_doorman</th>
      <th>has_elevator</th>
      <th>has_dishwasher</th>
      <th>has_patio</th>
      <th>has_gym</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5000.000000</td>
      <td>5.000000e+03</td>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.00000</td>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5526.909400</td>
      <td>5.122007e+07</td>
      <td>4536.920800</td>
      <td>1.395700</td>
      <td>1.321600</td>
      <td>920.101400</td>
      <td>5.079200</td>
      <td>10.190200</td>
      <td>52.093200</td>
      <td>0.429600</td>
      <td>0.12860</td>
      <td>0.133800</td>
      <td>0.228000</td>
      <td>0.240000</td>
      <td>0.155600</td>
      <td>0.045600</td>
      <td>0.143800</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3263.692417</td>
      <td>2.802283e+07</td>
      <td>2929.838953</td>
      <td>0.961018</td>
      <td>0.565542</td>
      <td>440.150464</td>
      <td>5.268897</td>
      <td>10.565361</td>
      <td>40.224501</td>
      <td>0.495069</td>
      <td>0.33479</td>
      <td>0.340471</td>
      <td>0.419585</td>
      <td>0.427126</td>
      <td>0.362512</td>
      <td>0.208637</td>
      <td>0.350922</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>7.107000e+03</td>
      <td>1250.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>250.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2699.750000</td>
      <td>2.699811e+07</td>
      <td>2750.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>633.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>12.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5456.500000</td>
      <td>5.069894e+07</td>
      <td>3600.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>800.000000</td>
      <td>4.000000</td>
      <td>6.000000</td>
      <td>44.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>8306.000000</td>
      <td>7.572064e+07</td>
      <td>5200.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>1094.000000</td>
      <td>6.000000</td>
      <td>14.000000</td>
      <td>89.000000</td>
      <td>1.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>11349.000000</td>
      <td>9.998721e+07</td>
      <td>20000.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>4800.000000</td>
      <td>51.000000</td>
      <td>83.000000</td>
      <td>180.000000</td>
      <td>1.000000</td>
      <td>1.00000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.shape
```




    (5000, 20)




```python
data.columns
```




    Index(['rental_id', 'building_id', 'rent', 'bedrooms', 'bathrooms',
           'size_sqft', 'min_to_subway', 'floor', 'building_age_yrs', 'no_fee',
           'has_roofdeck', 'has_washer_dryer', 'has_doorman', 'has_elevator',
           'has_dishwasher', 'has_patio', 'has_gym', 'neighborhood', 'submarket',
           'borough'],
          dtype='object')



**Step 2**


```python
Y=data['rent']
# X for step 3,4,final
X=data.drop(columns=['rent','rental_id','neighborhood','building_id'])
# X_droped for step 2
X_droped= X.drop(columns=['submarket','borough'])

X.shape


```




    (5000, 16)




```python
X.head(2)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>size_sqft</th>
      <th>min_to_subway</th>
      <th>floor</th>
      <th>building_age_yrs</th>
      <th>no_fee</th>
      <th>has_roofdeck</th>
      <th>has_washer_dryer</th>
      <th>has_doorman</th>
      <th>has_elevator</th>
      <th>has_dishwasher</th>
      <th>has_patio</th>
      <th>has_gym</th>
      <th>submarket</th>
      <th>borough</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>1</td>
      <td>480</td>
      <td>9</td>
      <td>2.0</td>
      <td>17</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>All Upper East Side</td>
      <td>Manhattan</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>2</td>
      <td>2000</td>
      <td>4</td>
      <td>1.0</td>
      <td>96</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>All Downtown</td>
      <td>Manhattan</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_train, X_test, y_train, y_test = train_test_split(X_droped,Y, test_size=0.2)
model = LinearRegression(normalize=True)
model.fit(X_train, y_train)
y_predict=model.predict(X_test)
r2_score(y_test, y_predict)
```




    0.7343946046073938




```python
X.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>size_sqft</th>
      <th>min_to_subway</th>
      <th>floor</th>
      <th>building_age_yrs</th>
      <th>no_fee</th>
      <th>has_roofdeck</th>
      <th>has_washer_dryer</th>
      <th>has_doorman</th>
      <th>has_elevator</th>
      <th>has_dishwasher</th>
      <th>has_patio</th>
      <th>has_gym</th>
      <th>submarket</th>
      <th>borough</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>1</td>
      <td>480</td>
      <td>9</td>
      <td>2.0</td>
      <td>17</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>All Upper East Side</td>
      <td>Manhattan</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>2</td>
      <td>2000</td>
      <td>4</td>
      <td>1.0</td>
      <td>96</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>All Downtown</td>
      <td>Manhattan</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>1</td>
      <td>1000</td>
      <td>4</td>
      <td>1.0</td>
      <td>106</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Northwest Queens</td>
      <td>Queens</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>1</td>
      <td>916</td>
      <td>2</td>
      <td>51.0</td>
      <td>29</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>All Midtown</td>
      <td>Manhattan</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>1</td>
      <td>975</td>
      <td>3</td>
      <td>8.0</td>
      <td>31</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>All Downtown</td>
      <td>Manhattan</td>
    </tr>
  </tbody>
</table>
</div>



**Step 3**


```python
X_dummies = pd.get_dummies(X,columns=['submarket','borough'])
X_dummies.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>size_sqft</th>
      <th>min_to_subway</th>
      <th>floor</th>
      <th>building_age_yrs</th>
      <th>no_fee</th>
      <th>has_roofdeck</th>
      <th>has_washer_dryer</th>
      <th>has_doorman</th>
      <th>has_elevator</th>
      <th>has_dishwasher</th>
      <th>has_patio</th>
      <th>has_gym</th>
      <th>submarket_All Downtown</th>
      <th>submarket_All Midtown</th>
      <th>submarket_All Upper East Side</th>
      <th>submarket_All Upper Manhattan</th>
      <th>submarket_All Upper West Side</th>
      <th>submarket_Central Queens</th>
      <th>submarket_East Brooklyn</th>
      <th>submarket_North Brooklyn</th>
      <th>submarket_Northeast Queens</th>
      <th>submarket_Northwest Brooklyn</th>
      <th>submarket_Northwest Queens</th>
      <th>submarket_Prospect Park</th>
      <th>submarket_South Brooklyn</th>
      <th>submarket_South Queens</th>
      <th>submarket_The Rockaways</th>
      <th>borough_Brooklyn</th>
      <th>borough_Manhattan</th>
      <th>borough_Queens</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>1</td>
      <td>480</td>
      <td>9</td>
      <td>2.0</td>
      <td>17</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>2</td>
      <td>2000</td>
      <td>4</td>
      <td>1.0</td>
      <td>96</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>1</td>
      <td>1000</td>
      <td>4</td>
      <td>1.0</td>
      <td>106</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>1</td>
      <td>916</td>
      <td>2</td>
      <td>51.0</td>
      <td>29</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>1</td>
      <td>975</td>
      <td>3</td>
      <td>8.0</td>
      <td>31</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_train, X_test, y_train, y_test = train_test_split(X_dummies,Y, test_size=0.2)
model = LinearRegression(normalize=False)
model.fit(X_train, y_train)
y_predict=model.predict(X_test)
r2_score(y_test, y_predict)
```




    0.80168339311738



# step 4


```python
sns.distplot(Y)
plt.show()
```

![image](https://github.com/Zejabati/HousePricePrediction/assets/65095428/6974d82e-318a-442d-8bcb-1cbdbe44a01d)
    

```python
Y_log = np.log(Y)
sns.distplot(Y_log)
plt.show()

```

![image](https://github.com/Zejabati/HousePricePrediction/assets/65095428/ef781a92-c29d-4bb0-9b35-4684159d50b0)

    



```python
X_train, X_test, y_train, y_test = train_test_split(X_dummies,Y_log, test_size=0.2)
model = LinearRegression(normalize=True)
model.fit(X_train, y_train)
y_predict=model.predict(X_test)
r2_score(y_test, y_predict)
```




    -1.3267526086771694e+20




```python
X_train, X_test, y_train, y_test = train_test_split(X_dummies,Y_log, test_size=0.2)
model = LinearRegression(normalize=False)
model.fit(X_train, y_train)
y_predict=model.predict(X_test)
r2_score(y_test, y_predict)
```




    0.8423159048711364




```python
sns.distplot(1/Y)
plt.show()
```

![image](https://github.com/Zejabati/HousePricePrediction/assets/65095428/8eddc208-5917-468f-a740-60c39714bc41)
    



```python
X_train, X_test, y_train, y_test = train_test_split(X_dummies,1/Y, test_size=0.2)
model = LinearRegression(normalize=False)
model.fit(X_train, y_train)
y_predict=model.predict(X_test)
r2_score(y_test, y_predict)
```




    0.8191719926366454




```python
X_train, X_test, y_train, y_test = train_test_split(X_dummies,1/Y, test_size=0.2)
model = LinearRegression(normalize=True)
model.fit(X_train, y_train)
y_predict=model.predict(X_test)
r2_score(y_test, y_predict)
```




    0.7992418417772884




```python
score = []

for i in np.arange(0, 100, 1):
    X_train, X_test, Y_train, Y_test = train_test_split(X_dummies,Y_log, test_size=0.2)
    reg = LinearRegression(normalize=False)
    reg.fit(X_train, Y_train)
    r2_score(Y_test, reg.predict(X_test))
    score.append(r2_score(Y_test, reg.predict(X_test)))

np.mean(score)

```




    0.854028216609156



**High Correlation Filter**


```python
corr= X.corr()
corr
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>size_sqft</th>
      <th>min_to_subway</th>
      <th>floor</th>
      <th>building_age_yrs</th>
      <th>no_fee</th>
      <th>has_roofdeck</th>
      <th>has_washer_dryer</th>
      <th>has_doorman</th>
      <th>has_elevator</th>
      <th>has_dishwasher</th>
      <th>has_patio</th>
      <th>has_gym</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bedrooms</th>
      <td>1.000000</td>
      <td>0.647499</td>
      <td>0.738410</td>
      <td>0.053365</td>
      <td>-0.006468</td>
      <td>0.070329</td>
      <td>-0.067047</td>
      <td>-0.012395</td>
      <td>-0.002582</td>
      <td>-0.034031</td>
      <td>-0.027457</td>
      <td>-0.007094</td>
      <td>0.003772</td>
      <td>-0.018393</td>
    </tr>
    <tr>
      <th>bathrooms</th>
      <td>0.647499</td>
      <td>1.000000</td>
      <td>0.770593</td>
      <td>0.054891</td>
      <td>0.145303</td>
      <td>-0.103050</td>
      <td>-0.052726</td>
      <td>0.044597</td>
      <td>0.052827</td>
      <td>0.044997</td>
      <td>0.051410</td>
      <td>0.060295</td>
      <td>0.057091</td>
      <td>0.057221</td>
    </tr>
    <tr>
      <th>size_sqft</th>
      <td>0.738410</td>
      <td>0.770593</td>
      <td>1.000000</td>
      <td>0.024422</td>
      <td>0.100234</td>
      <td>0.026434</td>
      <td>-0.129534</td>
      <td>0.038302</td>
      <td>0.052746</td>
      <td>0.046402</td>
      <td>0.060830</td>
      <td>0.063051</td>
      <td>0.036600</td>
      <td>0.047815</td>
    </tr>
    <tr>
      <th>min_to_subway</th>
      <td>0.053365</td>
      <td>0.054891</td>
      <td>0.024422</td>
      <td>1.000000</td>
      <td>0.029242</td>
      <td>-0.138273</td>
      <td>0.054977</td>
      <td>-0.021765</td>
      <td>-0.017840</td>
      <td>-0.006722</td>
      <td>-0.014314</td>
      <td>-0.022058</td>
      <td>-0.005652</td>
      <td>-0.014924</td>
    </tr>
    <tr>
      <th>floor</th>
      <td>-0.006468</td>
      <td>0.145303</td>
      <td>0.100234</td>
      <td>0.029242</td>
      <td>1.000000</td>
      <td>-0.352757</td>
      <td>0.088094</td>
      <td>0.079554</td>
      <td>0.060379</td>
      <td>0.123108</td>
      <td>0.109258</td>
      <td>0.036979</td>
      <td>0.068119</td>
      <td>0.089361</td>
    </tr>
    <tr>
      <th>building_age_yrs</th>
      <td>0.070329</td>
      <td>-0.103050</td>
      <td>0.026434</td>
      <td>-0.138273</td>
      <td>-0.352757</td>
      <td>1.000000</td>
      <td>-0.213815</td>
      <td>-0.055064</td>
      <td>-0.031511</td>
      <td>-0.058163</td>
      <td>-0.067261</td>
      <td>-0.040078</td>
      <td>-0.048155</td>
      <td>-0.063687</td>
    </tr>
    <tr>
      <th>no_fee</th>
      <td>-0.067047</td>
      <td>-0.052726</td>
      <td>-0.129534</td>
      <td>0.054977</td>
      <td>0.088094</td>
      <td>-0.213815</td>
      <td>1.000000</td>
      <td>-0.089593</td>
      <td>-0.082366</td>
      <td>-0.175022</td>
      <td>-0.155638</td>
      <td>-0.086081</td>
      <td>-0.052192</td>
      <td>-0.105797</td>
    </tr>
    <tr>
      <th>has_roofdeck</th>
      <td>-0.012395</td>
      <td>0.044597</td>
      <td>0.038302</td>
      <td>-0.021765</td>
      <td>0.079554</td>
      <td>-0.055064</td>
      <td>-0.089593</td>
      <td>1.000000</td>
      <td>0.331626</td>
      <td>0.506101</td>
      <td>0.535333</td>
      <td>0.369123</td>
      <td>0.145139</td>
      <td>0.579826</td>
    </tr>
    <tr>
      <th>has_washer_dryer</th>
      <td>-0.002582</td>
      <td>0.052827</td>
      <td>0.052746</td>
      <td>-0.017840</td>
      <td>0.060379</td>
      <td>-0.031511</td>
      <td>-0.082366</td>
      <td>0.331626</td>
      <td>1.000000</td>
      <td>0.356328</td>
      <td>0.410523</td>
      <td>0.471480</td>
      <td>0.159091</td>
      <td>0.376372</td>
    </tr>
    <tr>
      <th>has_doorman</th>
      <td>-0.034031</td>
      <td>0.044997</td>
      <td>0.046402</td>
      <td>-0.006722</td>
      <td>0.123108</td>
      <td>-0.058163</td>
      <td>-0.175022</td>
      <td>0.506101</td>
      <td>0.356328</td>
      <td>1.000000</td>
      <td>0.728208</td>
      <td>0.374312</td>
      <td>0.157709</td>
      <td>0.648138</td>
    </tr>
    <tr>
      <th>has_elevator</th>
      <td>-0.027457</td>
      <td>0.051410</td>
      <td>0.060830</td>
      <td>-0.014314</td>
      <td>0.109258</td>
      <td>-0.067261</td>
      <td>-0.155638</td>
      <td>0.535333</td>
      <td>0.410523</td>
      <td>0.728208</td>
      <td>1.000000</td>
      <td>0.451245</td>
      <td>0.151027</td>
      <td>0.659879</td>
    </tr>
    <tr>
      <th>has_dishwasher</th>
      <td>-0.007094</td>
      <td>0.060295</td>
      <td>0.063051</td>
      <td>-0.022058</td>
      <td>0.036979</td>
      <td>-0.040078</td>
      <td>-0.086081</td>
      <td>0.369123</td>
      <td>0.471480</td>
      <td>0.374312</td>
      <td>0.451245</td>
      <td>1.000000</td>
      <td>0.154786</td>
      <td>0.379161</td>
    </tr>
    <tr>
      <th>has_patio</th>
      <td>0.003772</td>
      <td>0.057091</td>
      <td>0.036600</td>
      <td>-0.005652</td>
      <td>0.068119</td>
      <td>-0.048155</td>
      <td>-0.052192</td>
      <td>0.145139</td>
      <td>0.159091</td>
      <td>0.157709</td>
      <td>0.151027</td>
      <td>0.154786</td>
      <td>1.000000</td>
      <td>0.150856</td>
    </tr>
    <tr>
      <th>has_gym</th>
      <td>-0.018393</td>
      <td>0.057221</td>
      <td>0.047815</td>
      <td>-0.014924</td>
      <td>0.089361</td>
      <td>-0.063687</td>
      <td>-0.105797</td>
      <td>0.579826</td>
      <td>0.376372</td>
      <td>0.648138</td>
      <td>0.659879</td>
      <td>0.379161</td>
      <td>0.150856</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize = (10,10))
sns.heatmap(corr, vmin=-1, vmax=1)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f03534b6fd0>



![image](https://github.com/Zejabati/HousePricePrediction/assets/65095428/ae925e32-2a88-4829-9b27-7655e740051e)
    



```python
X.columns
```




    Index(['bedrooms', 'bathrooms', 'size_sqft', 'min_to_subway', 'floor',
           'building_age_yrs', 'no_fee', 'has_roofdeck', 'has_washer_dryer',
           'has_doorman', 'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym',
           'submarket', 'borough'],
          dtype='object')




```python
# Create correlation matrix
corr_matrix = X.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find features with correlation greater than 0.75
to_drop = [column for column in upper.columns if any(upper[column] > 0.75)]

# Drop features/ Xcorr: x after High Correlation Filter
Xcorr=X.drop(to_drop, axis=1, inplace=False)
Xcorr.columns
```




    Index(['bedrooms', 'bathrooms', 'min_to_subway', 'floor', 'building_age_yrs',
           'no_fee', 'has_roofdeck', 'has_washer_dryer', 'has_doorman',
           'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym', 'submarket',
           'borough'],
          dtype='object')




```python
#X_dummiescorr: x after High Correlation Filter and using get_dummies
X_dummiescorr = pd.get_dummies(Xcorr,columns=['submarket','borough'])

score = []

for i in np.arange(0, 100, 1):
    X_train, X_test, Y_train, Y_test = train_test_split(X_dummiescorr,Y_log, test_size=0.2)
    reg = LinearRegression(normalize=False)
    reg.fit(X_train, Y_train)
    r2_score(Y_test, reg.predict(X_test))
    score.append(r2_score(Y_test, reg.predict(X_test)))

np.mean(score)

```




    0.7692567725614283



**ICA**


```python
X_dummies.shape
```




    (5000, 32)




```python
transformer = FastICA(n_components=28,random_state=0)
# X_transformed: dimensionality reduction using ICA method for X_dummies
X_transformed = transformer.fit_transform(X_dummies)
X_transformed.shape
```




    (5000, 28)




```python
score = []

for i in np.arange(0, 100, 1):
    X_train, X_test, Y_train, Y_test = train_test_split(X_transformed,Y_log, test_size=0.2)
    reg = LinearRegression(normalize=False)
    reg.fit(X_train, Y_train)
    r2_score(Y_test, reg.predict(X_test))
    score.append(r2_score(Y_test, reg.predict(X_test)))

np.mean(score)
```




    0.85454989520325



**UMAP**


```python
X1=data.drop(columns=['rent','rental_id','building_id'])
X1_dummies = pd.get_dummies(X1,columns=['submarket','borough','neighborhood'])
X1_dummies.shape
```




    (5000, 125)




```python
reducer = umap.UMAP(n_components=110)
embedding = reducer.fit_transform(X1_dummies)
embedding.shape
```




    (5000, 110)




```python
score = []

for i in np.arange(0, 100, 1):
    X_train, X_test, Y_train, Y_test = train_test_split(embedding,Y_log, test_size=0.2)
    reg = LinearRegression(normalize=False)
    reg.fit(X_train, Y_train)
    r2_score(Y_test, reg.predict(X_test))
    score.append(r2_score(Y_test, reg.predict(X_test)))

np.mean(score)
```




    0.6574575565530977



**Ridge**


```python
X_train, X_test, y_train, y_test = train_test_split(X_dummies,Y_log, test_size=0.2)
X_train=preprocessing.scale(X_train)
X_test=preprocessing.scale(X_test)
y_train=preprocessing.scale(y_train)
y_test=preprocessing.scale(y_test)

rr = Ridge(alpha=0.01,normalize=False)
rr.fit(X_train, y_train)
pred_train_rr= rr.predict(X_train)
print('MSE_train:',np.sqrt(mean_squared_error(y_train,pred_train_rr)))
print('r2_score_train:',r2_score(y_train, pred_train_rr))

pred_test_rr= rr.predict(X_test)
print('MSE_test:',np.sqrt(mean_squared_error(y_test,pred_test_rr)))
print('r2_score_test:',r2_score(y_test, pred_test_rr))


```

    MSE_train: 0.3791018984561559
    r2_score_train: 0.8562817505869385
    MSE_test: 0.3718145267822194
    r2_score_test: 0.8617539576737143


 **Lasso**


```python
model_lasso = Lasso(alpha=0.01,normalize=False)
model_lasso.fit(X_train, y_train)
pred_train_lasso= model_lasso.predict(X_train)
print('MSE_train:',np.sqrt(mean_squared_error(y_train,pred_train_lasso)))
print('r2_score_train:',r2_score(y_train, pred_train_lasso))

pred_test_lasso= model_lasso.predict(X_test)
print('MSE_test:',np.sqrt(mean_squared_error(y_test,pred_test_lasso)))
print('r2_score_test:',r2_score(y_test, pred_test_lasso))
```

    MSE_train: 0.3815272061201945
    r2_score_train: 0.8544369909901187
    MSE_test: 0.3737393941516346
    r2_score_test: 0.8603188652591691


**Elastic** **Net**


```python
model_enet = ElasticNet(alpha = 0.01, normalize=False)
model_enet.fit(X_train, y_train)
pred_train_enet= model_enet.predict(X_train)
print('MSE_train:',np.sqrt(mean_squared_error(y_train,pred_train_enet)))
print('r2_score_train:',r2_score(y_train, pred_train_enet))

pred_test_enet= model_enet.predict(X_test)
print('MSE_test:',np.sqrt(mean_squared_error(y_test,pred_test_enet)))
print('r2_score_test:',r2_score(y_test, pred_test_enet))
```

    MSE_train: 0.380124346674705
    r2_score_train: 0.8555054810651287
    MSE_test: 0.3722495439869019
    r2_score_test: 0.8614302770015436


