# <center>Uber Analysis</center>

# 1.Data Import


```python
### lets import all the necessary packages !

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```


```python
import os
```


```python
os.listdir("C:/Users/Dell/Downloads/UberAnalysis/Datasets")
```




    ['other-American_B01362.csv',
     'other-Carmel_B00256.csv',
     'other-Dial7_B00887.csv',
     'other-Diplo_B01196.csv',
     'other-Federal_02216.csv',
     'other-FHV-services_jan-aug-2015.csv',
     'other-Firstclass_B01536.csv',
     'other-Highclass_B01717.csv',
     'other-Lyft_B02510.csv',
     'other-Prestige_B01338.csv',
     'other-Skyline_B00111.csv',
     'Uber-Jan-Feb-FOIL.csv',
     'uber-raw-data-apr14.csv',
     'uber-raw-data-aug14.csv',
     'uber-raw-data-janjune-15.csv',
     'uber-raw-data-janjune-15_sample.csv',
     'uber-raw-data-jul14.csv',
     'uber-raw-data-jun14.csv',
     'uber-raw-data-may14.csv',
     'uber-raw-data-sep14.csv']




```python
uber_15 = pd.read_csv("C:/Users/Dell/Downloads/UberAnalysis/Datasets/uber-raw-data-janjune-15_sample.csv")
```


```python
#data shape
uber_15.shape
```




    (100000, 4)



## 2. Data Preprocessing
        check data-type , check missing values , check whether duplicated values or not !
        ie Prepare Data for Analysis !


```python
#check types
type(uber_15)
```




    pandas.core.frame.DataFrame



**a) Handle duplicates**


```python
uber_15.duplicated().sum()
```




    54




```python
uber_15.drop_duplicates(inplace=True)
```


```python
uber_15.duplicated().sum()
```




    0



**b) Check missing values**


```python
uber_15.isnull().sum()
```




    Dispatching_base_num       0
    Pickup_date                0
    Affiliated_base_num     1116
    locationID                 0
    dtype: int64




```python
uber_15['Affiliated_base_num']
```




    0        B02764
    1        B02682
    2        B02617
    3        B02764
    4        B00111
              ...  
    99995    B02764
    99996    B02764
    99997    B02598
    99998    B02682
    99999    B02764
    Name: Affiliated_base_num, Length: 99946, dtype: object



**c) Adjust dtypes**


```python
uber_15.shape
```




    (99946, 4)




```python
uber_15.dtypes
```




    Dispatching_base_num    object
    Pickup_date             object
    Affiliated_base_num     object
    locationID               int64
    dtype: object




```python
uber_15['Pickup_date'][0]
```




    '2015-05-02 21:43:00'




```python
type(uber_15['Pickup_date'][0])
```




    str




```python
uber_15['Pickup_date'] = pd.to_datetime(uber_15['Pickup_date'])
```


```python
uber_15['Pickup_date'].dtype
```




    dtype('<M8[ns]')




```python
uber_15['Pickup_date'][0]
```




    Timestamp('2015-05-02 21:43:00')




```python
type(uber_15['Pickup_date'][0])
```




    pandas._libs.tslibs.timestamps.Timestamp




```python
uber_15.dtypes
```




    Dispatching_base_num            object
    Pickup_date             datetime64[ns]
    Affiliated_base_num             object
    locationID                       int64
    dtype: object



# 3. Data Exploration

**a) Uber Runs by month**


```python
uber_15.head()
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
      <th>Dispatching_base_num</th>
      <th>Pickup_date</th>
      <th>Affiliated_base_num</th>
      <th>locationID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>B02617</td>
      <td>2015-05-02 21:43:00</td>
      <td>B02764</td>
      <td>237</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B02682</td>
      <td>2015-01-20 19:52:59</td>
      <td>B02682</td>
      <td>231</td>
    </tr>
    <tr>
      <th>2</th>
      <td>B02617</td>
      <td>2015-03-19 20:26:00</td>
      <td>B02617</td>
      <td>161</td>
    </tr>
    <tr>
      <th>3</th>
      <td>B02764</td>
      <td>2015-04-10 17:38:00</td>
      <td>B02764</td>
      <td>107</td>
    </tr>
    <tr>
      <th>4</th>
      <td>B02764</td>
      <td>2015-03-23 07:03:00</td>
      <td>B00111</td>
      <td>140</td>
    </tr>
  </tbody>
</table>
</div>




```python
uber_15['month'] = uber_15['Pickup_date'].dt.month_name()
```


```python
uber_15['month']
```




    0            May
    1        January
    2          March
    3          April
    4          March
              ...   
    99995      April
    99996      March
    99997      March
    99998        May
    99999       June
    Name: month, Length: 99946, dtype: object




```python
uber_15['month'].value_counts().plot(kind='bar')

plt.xlabel("Month")
plt.ylabel("Total pickups in NYC")

plt.show()
```


    
![png](output_31_0.png)
    


June has the max pickup while January has the minimum.

**b) Total pickups by weekday per month**


```python
## extracting dervied features (weekday ,day ,hour ,month ,minute) from 'Pickup_date'..

uber_15['weekday'] = uber_15['Pickup_date'].dt.day_name()
uber_15['day'] = uber_15['Pickup_date'].dt.day
uber_15['hour'] = uber_15['Pickup_date'].dt.hour
uber_15['minute'] = uber_15['Pickup_date'].dt.minute
```


```python
uber_15.head()
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
      <th>Dispatching_base_num</th>
      <th>Pickup_date</th>
      <th>Affiliated_base_num</th>
      <th>locationID</th>
      <th>month</th>
      <th>weekday</th>
      <th>day</th>
      <th>hour</th>
      <th>minute</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>B02617</td>
      <td>2015-05-02 21:43:00</td>
      <td>B02764</td>
      <td>237</td>
      <td>May</td>
      <td>Saturday</td>
      <td>2</td>
      <td>21</td>
      <td>43</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B02682</td>
      <td>2015-01-20 19:52:59</td>
      <td>B02682</td>
      <td>231</td>
      <td>January</td>
      <td>Tuesday</td>
      <td>20</td>
      <td>19</td>
      <td>52</td>
    </tr>
    <tr>
      <th>2</th>
      <td>B02617</td>
      <td>2015-03-19 20:26:00</td>
      <td>B02617</td>
      <td>161</td>
      <td>March</td>
      <td>Thursday</td>
      <td>19</td>
      <td>20</td>
      <td>26</td>
    </tr>
    <tr>
      <th>3</th>
      <td>B02764</td>
      <td>2015-04-10 17:38:00</td>
      <td>B02764</td>
      <td>107</td>
      <td>April</td>
      <td>Friday</td>
      <td>10</td>
      <td>17</td>
      <td>38</td>
    </tr>
    <tr>
      <th>4</th>
      <td>B02764</td>
      <td>2015-03-23 07:03:00</td>
      <td>B00111</td>
      <td>140</td>
      <td>March</td>
      <td>Monday</td>
      <td>23</td>
      <td>7</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
## pivot table

pivot = pd.crosstab(index=uber_15['month'] , columns=uber_15['weekday'])
```


```python
pivot
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
      <th>weekday</th>
      <th>Friday</th>
      <th>Monday</th>
      <th>Saturday</th>
      <th>Sunday</th>
      <th>Thursday</th>
      <th>Tuesday</th>
      <th>Wednesday</th>
    </tr>
    <tr>
      <th>month</th>
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
      <th>April</th>
      <td>2365</td>
      <td>1833</td>
      <td>2508</td>
      <td>2052</td>
      <td>2823</td>
      <td>1880</td>
      <td>2521</td>
    </tr>
    <tr>
      <th>February</th>
      <td>2655</td>
      <td>1970</td>
      <td>2550</td>
      <td>2183</td>
      <td>2396</td>
      <td>2129</td>
      <td>2013</td>
    </tr>
    <tr>
      <th>January</th>
      <td>2508</td>
      <td>1353</td>
      <td>2745</td>
      <td>1651</td>
      <td>2378</td>
      <td>1444</td>
      <td>1740</td>
    </tr>
    <tr>
      <th>June</th>
      <td>2793</td>
      <td>2848</td>
      <td>3037</td>
      <td>2485</td>
      <td>2767</td>
      <td>3187</td>
      <td>2503</td>
    </tr>
    <tr>
      <th>March</th>
      <td>2465</td>
      <td>2115</td>
      <td>2522</td>
      <td>2379</td>
      <td>2093</td>
      <td>2388</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>May</th>
      <td>3262</td>
      <td>1865</td>
      <td>3519</td>
      <td>2944</td>
      <td>2627</td>
      <td>2115</td>
      <td>2328</td>
    </tr>
  </tbody>
</table>
</div>




```python
## grouped-bar plot using Pandas ..
pivot.plot(kind='bar' , figsize=(8,6))
```




    <AxesSubplot:xlabel='month'>




    
![png](output_38_1.png)
    


On Saturday & Friday, u are getting more Uber pickups in each month , it seems that New Yorkers used to go for 
shopping , Malls , fun activities alot on these days

# 4. Rush hours on NYC


```python
summary = uber_15.groupby(['weekday' , 'hour'] , as_index=False).size()
```


```python
summary
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
      <th>weekday</th>
      <th>hour</th>
      <th>size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Friday</td>
      <td>0</td>
      <td>581</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Friday</td>
      <td>1</td>
      <td>333</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Friday</td>
      <td>2</td>
      <td>197</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Friday</td>
      <td>3</td>
      <td>138</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Friday</td>
      <td>4</td>
      <td>161</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>163</th>
      <td>Wednesday</td>
      <td>19</td>
      <td>1044</td>
    </tr>
    <tr>
      <th>164</th>
      <td>Wednesday</td>
      <td>20</td>
      <td>897</td>
    </tr>
    <tr>
      <th>165</th>
      <td>Wednesday</td>
      <td>21</td>
      <td>949</td>
    </tr>
    <tr>
      <th>166</th>
      <td>Wednesday</td>
      <td>22</td>
      <td>900</td>
    </tr>
    <tr>
      <th>167</th>
      <td>Wednesday</td>
      <td>23</td>
      <td>669</td>
    </tr>
  </tbody>
</table>
<p>168 rows × 3 columns</p>
</div>




```python
## pointplot between 'hour' & 'size' for all the weekdays..

plt.figure(figsize=(8,6))
sns.pointplot(x="hour" , y="size" , hue="weekday" , data=summary)
```




    <AxesSubplot:xlabel='hour', ylabel='size'>




    
![png](output_43_1.png)
    


It's interesting to see that Saturday and Sunday exhibit similar demand throughout the late night/morning/afternoon, 
but it exhibits opposite trends during the evening. In the evening, Saturday pickups continue to increase throughout the evening,
but Sunday pickups takes a downward turn after evening..

We can see that there the weekdays that has the most demand during the late evening is Friday and Saturday, 
which is expected, but what strikes me is that Thursday nights also exhibits very similar trends as Friday and Saturday nights.

It seems like New Yorkers are starting their 'weekends' on Thursday nights. :)

# 5. Active vehicles by base_num


```python
uber_15.columns
```




    Index(['Dispatching_base_num', 'Pickup_date', 'Affiliated_base_num',
           'locationID', 'month', 'weekday', 'day', 'hour', 'minute'],
          dtype='object')




```python
uber_foil = pd.read_csv("C:/Users/Dell/Downloads/UberAnalysis/Datasets/Uber-Jan-Feb-FOIL.csv")
uber_foil.head()
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
      <th>dispatching_base_number</th>
      <th>date</th>
      <th>active_vehicles</th>
      <th>trips</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>B02512</td>
      <td>1/1/2015</td>
      <td>190</td>
      <td>1132</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B02765</td>
      <td>1/1/2015</td>
      <td>225</td>
      <td>1765</td>
    </tr>
    <tr>
      <th>2</th>
      <td>B02764</td>
      <td>1/1/2015</td>
      <td>3427</td>
      <td>29421</td>
    </tr>
    <tr>
      <th>3</th>
      <td>B02682</td>
      <td>1/1/2015</td>
      <td>945</td>
      <td>7679</td>
    </tr>
    <tr>
      <th>4</th>
      <td>B02617</td>
      <td>1/1/2015</td>
      <td>1228</td>
      <td>9537</td>
    </tr>
  </tbody>
</table>
</div>




```python
uber_foil.shape
```




    (354, 4)




```python
!pip install chart_studio
```

    Collecting chart_studio
      Downloading chart_studio-1.1.0-py3-none-any.whl (64 kB)
         ---------------------------------------- 64.4/64.4 kB 3.4 MB/s eta 0:00:00
    Requirement already satisfied: requests in c:\users\dell\anaconda3\lib\site-packages (from chart_studio) (2.28.1)
    Requirement already satisfied: plotly in c:\users\dell\anaconda3\lib\site-packages (from chart_studio) (5.9.0)
    Collecting retrying>=1.3.3
      Downloading retrying-1.3.4-py3-none-any.whl (11 kB)
    Requirement already satisfied: six in c:\users\dell\anaconda3\lib\site-packages (from chart_studio) (1.16.0)
    Requirement already satisfied: tenacity>=6.2.0 in c:\users\dell\anaconda3\lib\site-packages (from plotly->chart_studio) (8.0.1)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in c:\users\dell\anaconda3\lib\site-packages (from requests->chart_studio) (1.26.11)
    Requirement already satisfied: charset-normalizer<3,>=2 in c:\users\dell\anaconda3\lib\site-packages (from requests->chart_studio) (2.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in c:\users\dell\anaconda3\lib\site-packages (from requests->chart_studio) (2022.9.14)
    Requirement already satisfied: idna<4,>=2.5 in c:\users\dell\anaconda3\lib\site-packages (from requests->chart_studio) (3.3)
    Installing collected packages: retrying, chart_studio
    Successfully installed chart_studio-1.1.0 retrying-1.3.4
    


```python
import chart_studio.plotly as py
import plotly.graph_objs as go
import plotly.express as px

from plotly.offline import download_plotlyjs , init_notebook_mode , plot , iplot 
```


```python
init_notebook_mode(connected=True)
```


<script type="text/javascript">
window.PlotlyConfig = {MathJaxConfig: 'local'};
if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
if (typeof require !== 'undefined') {
require.undef("plotly");
requirejs.config({
    paths: {
        'plotly': ['https://cdn.plot.ly/plotly-2.12.1.min']
    }
});
require(['plotly'], function(Plotly) {
    window._Plotly = Plotly;
});
}
</script>




```python
uber_foil.columns
```




    Index(['dispatching_base_number', 'date', 'active_vehicles', 'trips'], dtype='object')




```python
px.box(x='dispatching_base_number' , y='active_vehicles' , data_frame=uber_foil)
```


<div>                            <div id="0c8f244c-cf49-4555-9b4b-ccfa4dffafc7" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("0c8f244c-cf49-4555-9b4b-ccfa4dffafc7")) {                    Plotly.newPlot(                        "0c8f244c-cf49-4555-9b4b-ccfa4dffafc7",                        [{"alignmentgroup":"True","hovertemplate":"dispatching_base_number=%{x}<br>active_vehicles=%{y}<extra></extra>","legendgroup":"","marker":{"color":"#636efa"},"name":"","notched":false,"offsetgroup":"","orientation":"v","showlegend":false,"x":["B02512","B02765","B02764","B02682","B02617","B02598","B02598","B02617","B02512","B02682","B02765","B02764","B02765","B02617","B02598","B02682","B02512","B02764","B02512","B02682","B02598","B02765","B02617","B02764","B02512","B02682","B02617","B02764","B02598","B02765","B02764","B02682","B02617","B02765","B02512","B02598","B02617","B02682","B02764","B02765","B02512","B02598","B02765","B02598","B02512","B02682","B02764","B02617","B02617","B02512","B02764","B02682","B02598","B02765","B02682","B02617","B02598","B02512","B02764","B02765","B02765","B02598","B02682","B02764","B02617","B02512","B02764","B02765","B02512","B02598","B02682","B02617","B02765","B02598","B02617","B02512","B02764","B02682","B02764","B02765","B02598","B02512","B02617","B02682","B02512","B02682","B02617","B02765","B02764","B02598","B02617","B02765","B02764","B02682","B02512","B02598","B02598","B02512","B02682","B02765","B02617","B02764","B02512","B02598","B02765","B02764","B02682","B02617","B02682","B02617","B02765","B02764","B02512","B02598","B02598","B02682","B02512","B02764","B02765","B02617","B02764","B02512","B02682","B02598","B02765","B02617","B02617","B02764","B02512","B02598","B02682","B02765","B02598","B02512","B02765","B02764","B02617","B02682","B02598","B02764","B02512","B02617","B02682","B02765","B02512","B02764","B02765","B02598","B02682","B02617","B02617","B02598","B02765","B02764","B02682","B02512","B02682","B02765","B02617","B02598","B02512","B02764","B02764","B02682","B02765","B02617","B02598","B02512","B02617","B02764","B02682","B02765","B02512","B02598","B02512","B02617","B02682","B02764","B02765","B02598","B02765","B02512","B02617","B02682","B02764","B02598","B02598","B02682","B02512","B02765","B02617","B02764","B02617","B02682","B02765","B02598","B02764","B02512","B02765","B02598","B02512","B02764","B02617","B02682","B02764","B02765","B02512","B02682","B02617","B02598","B02617","B02682","B02598","B02512","B02765","B02764","B02617","B02765","B02598","B02512","B02764","B02682","B02598","B02512","B02617","B02682","B02764","B02765","B02764","B02765","B02598","B02617","B02682","B02512","B02617","B02682","B02598","B02764","B02512","B02765","B02764","B02512","B02617","B02765","B02682","B02598","B02617","B02764","B02512","B02598","B02765","B02682","B02617","B02512","B02682","B02765","B02598","B02764","B02617","B02682","B02764","B02765","B02512","B02598","B02764","B02512","B02598","B02765","B02617","B02682","B02682","B02764","B02617","B02765","B02512","B02598","B02598","B02512","B02617","B02764","B02682","B02765","B02764","B02512","B02682","B02617","B02598","B02765","B02598","B02682","B02617","B02765","B02764","B02512","B02598","B02512","B02682","B02764","B02765","B02617","B02764","B02617","B02598","B02682","B02765","B02512","B02598","B02682","B02765","B02617","B02512","B02764","B02512","B02617","B02682","B02764","B02598","B02765","B02598","B02617","B02764","B02682","B02765","B02512","B02764","B02512","B02598","B02682","B02617","B02765","B02512","B02598","B02765","B02682","B02617","B02764","B02598","B02617","B02682","B02512","B02765","B02764","B02765","B02617","B02598","B02512","B02764","B02682","B02598","B02764","B02617","B02682","B02512","B02765"],"x0":" ","xaxis":"x","y":[190,225,3427,945,1228,870,785,1137,175,890,196,3147,201,1188,818,915,173,3215,147,812,746,183,1088,2862,194,951,1218,3387,907,227,3473,1022,1336,234,218,933,1363,1039,3603,248,217,974,262,1070,238,1135,3831,1463,1455,224,3820,1140,1070,280,1057,1331,949,206,3558,245,220,832,943,3186,1228,162,3499,279,217,964,1082,1323,258,975,1342,234,3658,1092,3736,271,1030,233,1405,1174,237,1208,1457,270,3840,1068,1445,290,3975,1250,234,1079,974,201,1137,252,1306,3657,177,869,248,3290,1056,1223,883,992,238,2958,168,706,944,1151,221,3654,272,1350,3718,242,1228,1035,296,1429,1471,3889,246,1071,1295,295,1093,246,299,4040,1482,1330,945,3652,211,1367,1223,245,183,3300,226,829,1046,1203,1150,860,230,3012,1084,197,600,135,596,434,112,1619,3692,1235,286,1356,1011,235,1474,3959,1316,295,250,1082,256,1501,1384,4124,322,1106,309,225,1394,1321,3947,1027,961,1214,193,289,1355,3740,1217,1152,275,939,3270,227,299,991,257,3674,1350,1269,3856,309,244,1311,1393,1072,1524,1418,1179,264,355,4093,1526,385,1181,261,4170,1414,1031,211,1383,1300,3849,345,3422,313,923,1256,1136,176,1312,1241,976,3543,228,388,3700,233,1364,422,1281,1029,1450,3849,255,1115,450,1396,1532,269,1468,536,1181,4137,1590,1523,4395,599,281,1216,4129,236,1111,583,1486,1428,1261,3651,1293,521,210,1003,934,207,1214,3524,1164,508,3826,241,1314,1378,1066,578,1078,1314,1394,586,3842,228,1127,250,1428,4110,663,1452,4384,1574,1186,1497,736,272,1044,1374,685,1443,238,3981,199,1248,1220,3478,909,566,966,1332,3734,1262,665,238,3965,247,1061,1346,1456,698,246,1076,706,1395,1473,3934,1134,1539,1465,243,745,4101,786,1551,1114,272,4253,1510,994,3952,1372,1386,230,747],"y0":" ","yaxis":"y","type":"box"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"dispatching_base_number"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"active_vehicles"}},"legend":{"tracegroupgap":0},"margin":{"t":60},"boxmode":"group"},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('0c8f244c-cf49-4555-9b4b-ccfa4dffafc7');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



```python
px.violin(x='dispatching_base_number' , y='active_vehicles' , data_frame=uber_foil)
```


<div>                            <div id="fbbbaa68-0175-4ba3-b5ab-b35cf7d56093" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("fbbbaa68-0175-4ba3-b5ab-b35cf7d56093")) {                    Plotly.newPlot(                        "fbbbaa68-0175-4ba3-b5ab-b35cf7d56093",                        [{"alignmentgroup":"True","box":{"visible":false},"hovertemplate":"dispatching_base_number=%{x}<br>active_vehicles=%{y}<extra></extra>","legendgroup":"","marker":{"color":"#636efa"},"name":"","offsetgroup":"","orientation":"v","scalegroup":"True","showlegend":false,"x":["B02512","B02765","B02764","B02682","B02617","B02598","B02598","B02617","B02512","B02682","B02765","B02764","B02765","B02617","B02598","B02682","B02512","B02764","B02512","B02682","B02598","B02765","B02617","B02764","B02512","B02682","B02617","B02764","B02598","B02765","B02764","B02682","B02617","B02765","B02512","B02598","B02617","B02682","B02764","B02765","B02512","B02598","B02765","B02598","B02512","B02682","B02764","B02617","B02617","B02512","B02764","B02682","B02598","B02765","B02682","B02617","B02598","B02512","B02764","B02765","B02765","B02598","B02682","B02764","B02617","B02512","B02764","B02765","B02512","B02598","B02682","B02617","B02765","B02598","B02617","B02512","B02764","B02682","B02764","B02765","B02598","B02512","B02617","B02682","B02512","B02682","B02617","B02765","B02764","B02598","B02617","B02765","B02764","B02682","B02512","B02598","B02598","B02512","B02682","B02765","B02617","B02764","B02512","B02598","B02765","B02764","B02682","B02617","B02682","B02617","B02765","B02764","B02512","B02598","B02598","B02682","B02512","B02764","B02765","B02617","B02764","B02512","B02682","B02598","B02765","B02617","B02617","B02764","B02512","B02598","B02682","B02765","B02598","B02512","B02765","B02764","B02617","B02682","B02598","B02764","B02512","B02617","B02682","B02765","B02512","B02764","B02765","B02598","B02682","B02617","B02617","B02598","B02765","B02764","B02682","B02512","B02682","B02765","B02617","B02598","B02512","B02764","B02764","B02682","B02765","B02617","B02598","B02512","B02617","B02764","B02682","B02765","B02512","B02598","B02512","B02617","B02682","B02764","B02765","B02598","B02765","B02512","B02617","B02682","B02764","B02598","B02598","B02682","B02512","B02765","B02617","B02764","B02617","B02682","B02765","B02598","B02764","B02512","B02765","B02598","B02512","B02764","B02617","B02682","B02764","B02765","B02512","B02682","B02617","B02598","B02617","B02682","B02598","B02512","B02765","B02764","B02617","B02765","B02598","B02512","B02764","B02682","B02598","B02512","B02617","B02682","B02764","B02765","B02764","B02765","B02598","B02617","B02682","B02512","B02617","B02682","B02598","B02764","B02512","B02765","B02764","B02512","B02617","B02765","B02682","B02598","B02617","B02764","B02512","B02598","B02765","B02682","B02617","B02512","B02682","B02765","B02598","B02764","B02617","B02682","B02764","B02765","B02512","B02598","B02764","B02512","B02598","B02765","B02617","B02682","B02682","B02764","B02617","B02765","B02512","B02598","B02598","B02512","B02617","B02764","B02682","B02765","B02764","B02512","B02682","B02617","B02598","B02765","B02598","B02682","B02617","B02765","B02764","B02512","B02598","B02512","B02682","B02764","B02765","B02617","B02764","B02617","B02598","B02682","B02765","B02512","B02598","B02682","B02765","B02617","B02512","B02764","B02512","B02617","B02682","B02764","B02598","B02765","B02598","B02617","B02764","B02682","B02765","B02512","B02764","B02512","B02598","B02682","B02617","B02765","B02512","B02598","B02765","B02682","B02617","B02764","B02598","B02617","B02682","B02512","B02765","B02764","B02765","B02617","B02598","B02512","B02764","B02682","B02598","B02764","B02617","B02682","B02512","B02765"],"x0":" ","xaxis":"x","y":[190,225,3427,945,1228,870,785,1137,175,890,196,3147,201,1188,818,915,173,3215,147,812,746,183,1088,2862,194,951,1218,3387,907,227,3473,1022,1336,234,218,933,1363,1039,3603,248,217,974,262,1070,238,1135,3831,1463,1455,224,3820,1140,1070,280,1057,1331,949,206,3558,245,220,832,943,3186,1228,162,3499,279,217,964,1082,1323,258,975,1342,234,3658,1092,3736,271,1030,233,1405,1174,237,1208,1457,270,3840,1068,1445,290,3975,1250,234,1079,974,201,1137,252,1306,3657,177,869,248,3290,1056,1223,883,992,238,2958,168,706,944,1151,221,3654,272,1350,3718,242,1228,1035,296,1429,1471,3889,246,1071,1295,295,1093,246,299,4040,1482,1330,945,3652,211,1367,1223,245,183,3300,226,829,1046,1203,1150,860,230,3012,1084,197,600,135,596,434,112,1619,3692,1235,286,1356,1011,235,1474,3959,1316,295,250,1082,256,1501,1384,4124,322,1106,309,225,1394,1321,3947,1027,961,1214,193,289,1355,3740,1217,1152,275,939,3270,227,299,991,257,3674,1350,1269,3856,309,244,1311,1393,1072,1524,1418,1179,264,355,4093,1526,385,1181,261,4170,1414,1031,211,1383,1300,3849,345,3422,313,923,1256,1136,176,1312,1241,976,3543,228,388,3700,233,1364,422,1281,1029,1450,3849,255,1115,450,1396,1532,269,1468,536,1181,4137,1590,1523,4395,599,281,1216,4129,236,1111,583,1486,1428,1261,3651,1293,521,210,1003,934,207,1214,3524,1164,508,3826,241,1314,1378,1066,578,1078,1314,1394,586,3842,228,1127,250,1428,4110,663,1452,4384,1574,1186,1497,736,272,1044,1374,685,1443,238,3981,199,1248,1220,3478,909,566,966,1332,3734,1262,665,238,3965,247,1061,1346,1456,698,246,1076,706,1395,1473,3934,1134,1539,1465,243,745,4101,786,1551,1114,272,4253,1510,994,3952,1372,1386,230,747],"y0":" ","yaxis":"y","type":"violin"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"dispatching_base_number"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"active_vehicles"}},"legend":{"tracegroupgap":0},"margin":{"t":60},"violinmode":"group"},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('fbbbaa68-0175-4ba3-b5ab-b35cf7d56093');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


# 6.. Concat all datasets


```python
files = os.listdir("C:/Users/Dell/Downloads/UberAnalysis/Datasets")
files = files[-8:]
files.remove('uber-raw-data-janjune-15_sample.csv')
files.remove("uber-raw-data-janjune-15.csv")
files
```




    ['uber-raw-data-apr14.csv',
     'uber-raw-data-aug14.csv',
     'uber-raw-data-jul14.csv',
     'uber-raw-data-jun14.csv',
     'uber-raw-data-may14.csv',
     'uber-raw-data-sep14.csv']




```python
#blank dataframe
final = pd.DataFrame()

path = "C:/Users/Dell/Downloads/UberAnalysis/Datasets"

for file in files :
    print(path+'/'+file)
    current_df = pd.read_csv(path+'/'+file)
    final = pd.concat([current_df , final])
    
final.head()
```

    C:/Users/Dell/Downloads/UberAnalysis/Datasets/uber-raw-data-apr14.csv
    C:/Users/Dell/Downloads/UberAnalysis/Datasets/uber-raw-data-aug14.csv
    C:/Users/Dell/Downloads/UberAnalysis/Datasets/uber-raw-data-jul14.csv
    C:/Users/Dell/Downloads/UberAnalysis/Datasets/uber-raw-data-jun14.csv
    C:/Users/Dell/Downloads/UberAnalysis/Datasets/uber-raw-data-may14.csv
    C:/Users/Dell/Downloads/UberAnalysis/Datasets/uber-raw-data-sep14.csv
    




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
      <th>Date/Time</th>
      <th>Lat</th>
      <th>Lon</th>
      <th>Base</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9/1/2014 0:01:00</td>
      <td>40.2201</td>
      <td>-74.0021</td>
      <td>B02512</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9/1/2014 0:01:00</td>
      <td>40.7500</td>
      <td>-74.0027</td>
      <td>B02512</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9/1/2014 0:03:00</td>
      <td>40.7559</td>
      <td>-73.9864</td>
      <td>B02512</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9/1/2014 0:06:00</td>
      <td>40.7450</td>
      <td>-73.9889</td>
      <td>B02512</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9/1/2014 0:11:00</td>
      <td>40.8145</td>
      <td>-73.9444</td>
      <td>B02512</td>
    </tr>
  </tbody>
</table>
</div>




```python
final.shape
```




    (4534327, 4)




```python
### checkduplicate
final.duplicated().sum()
```




    82581




```python
## drop duplicate rows
final.drop_duplicates(inplace=True)
```


```python
final.shape
```




    (4451746, 4)




```python
final.duplicated().sum()
```




    0




```python
final.head()
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
      <th>Date/Time</th>
      <th>Lat</th>
      <th>Lon</th>
      <th>Base</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9/1/2014 0:01:00</td>
      <td>40.2201</td>
      <td>-74.0021</td>
      <td>B02512</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9/1/2014 0:01:00</td>
      <td>40.7500</td>
      <td>-74.0027</td>
      <td>B02512</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9/1/2014 0:03:00</td>
      <td>40.7559</td>
      <td>-73.9864</td>
      <td>B02512</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9/1/2014 0:06:00</td>
      <td>40.7450</td>
      <td>-73.9889</td>
      <td>B02512</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9/1/2014 0:11:00</td>
      <td>40.8145</td>
      <td>-73.9444</td>
      <td>B02512</td>
    </tr>
  </tbody>
</table>
</div>



## Dataset Information : 

### The dataset contains information about the Datetime, Latitude, Longitude and Base of each uber ride that happened in the month of July 2014 at New York City, USA

##### Date/Time : The date and time of the Uber pickup

##### Lat : The latitude of the Uber pickup

##### Lon : The longitude of the Uber pickup

##### Base : The TLC base company code affiliated with the Uber pickup

    The Base codes are for the following Uber bases:
    B02512 : Unter
    B02598 : Hinter
    B02617 : Weiter
    B02682 : Schmecken
    B02764 : Danach-NY


# 7. What locations of New York City we are getting rush ??


```python
rush_uber = final.groupby(['Lat' , 'Lon'] , as_index=False).size()
```


```python
rush_uber.head(6)
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
      <th>Lat</th>
      <th>Lon</th>
      <th>size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39.6569</td>
      <td>-74.2258</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>39.6686</td>
      <td>-74.1607</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>39.7214</td>
      <td>-74.2446</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>39.8416</td>
      <td>-74.1512</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>39.9055</td>
      <td>-74.0791</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>39.9196</td>
      <td>-74.1112</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
rush_uber.dtypes
```




    Lat     float64
    Lon     float64
    size      int64
    dtype: object




```python
rush_uber["size"] = rush_uber["size"].astype("int")
```


```python
rush_uber.dtypes
```




    Lat     float64
    Lon     float64
    size      int32
    dtype: object




```python
import folium
from folium.plugins import HeatMap
```


```python
basemap = folium.Map()

HeatMap(rush_uber).add_to(basemap)
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe srcdoc="&lt;!DOCTYPE html&gt;
&lt;html&gt;
&lt;head&gt;

    &lt;meta http-equiv=&quot;content-type&quot; content=&quot;text/html; charset=UTF-8&quot; /&gt;
    &lt;script&gt;L_PREFER_CANVAS = false; L_NO_TOUCH = false; L_DISABLE_3D = false;&lt;/script&gt;
    &lt;script src=&quot;https://cdn.jsdelivr.net/npm/leaflet@1.2.0/dist/leaflet.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://ajax.googleapis.com/ajax/libs/jquery/1.11.1/jquery.min.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://maxcdn.bootstrapcdn.com/bootstrap/3.2.0/js/bootstrap.min.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.js&quot;&gt;&lt;/script&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/leaflet@1.2.0/dist/leaflet.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap-theme.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://maxcdn.bootstrapcdn.com/font-awesome/4.6.3/css/font-awesome.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://rawgit.com/python-visualization/folium/master/folium/templates/leaflet.awesome.rotate.css&quot;/&gt;
    &lt;style&gt;html, body {width: 100%;height: 100%;margin: 0;padding: 0;}&lt;/style&gt;
    &lt;style&gt;#map {position:absolute;top:0;bottom:0;right:0;left:0;}&lt;/style&gt;

            &lt;style&gt; #map_b580f240724b33fd854a08e356488b6c {
                position : relative;
                width : 100.0%;
                height: 100.0%;
                left: 0.0%;
                top: 0.0%;
                }
            &lt;/style&gt;

&lt;/head&gt;
&lt;body&gt;


            &lt;div class=&quot;folium-map&quot; id=&quot;map_b580f240724b33fd854a08e356488b6c&quot; &gt;&lt;/div&gt;

&lt;/body&gt;
&lt;script&gt;




                var bounds = null;


            var map_b580f240724b33fd854a08e356488b6c = L.map(
                                  &#x27;map_b580f240724b33fd854a08e356488b6c&#x27;,
                                  {center: [0,0],
                                  zoom: 1,
                                  maxBounds: bounds,
                                  layers: [],
                                  worldCopyJump: false,
                                  crs: L.CRS.EPSG3857
                                 });



            var tile_layer_b2b1bf29ffa54dd733ce98e19f7d9bf6 = L.tileLayer(
                &#x27;https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png&#x27;,
                {
  &quot;attribution&quot;: null,
  &quot;detectRetina&quot;: false,
  &quot;maxZoom&quot;: 18,
  &quot;minZoom&quot;: 1,
  &quot;noWrap&quot;: false,
  &quot;subdomains&quot;: &quot;abc&quot;
}
                ).addTo(map_b580f240724b33fd854a08e356488b6c);

&lt;/script&gt;
&lt;/html&gt;" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



  We can see a number of hot spots here. Midtown Manhattan is clearly a huge bright spot
    & these are made from Midtown to Lower Manhattan followed by Upper Manhattan and the Heights of Brooklyn.

#  Examine rush on Hour and Weekday ( Perform Pair wise Analysis )


```python
final.columns
```




    Index(['Date/Time', 'Lat', 'Lon', 'Base'], dtype='object')




```python
final.head(3)
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
      <th>Date/Time</th>
      <th>Lat</th>
      <th>Lon</th>
      <th>Base</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9/1/2014 0:01:00</td>
      <td>40.2201</td>
      <td>-74.0021</td>
      <td>B02512</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9/1/2014 0:01:00</td>
      <td>40.7500</td>
      <td>-74.0027</td>
      <td>B02512</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9/1/2014 0:03:00</td>
      <td>40.7559</td>
      <td>-73.9864</td>
      <td>B02512</td>
    </tr>
  </tbody>
</table>
</div>




```python
final.dtypes
```




    Date/Time     object
    Lat          float64
    Lon          float64
    Base          object
    dtype: object




```python
final['Date/Time'][0]
```




    0    9/1/2014 0:01:00
    0    5/1/2014 0:02:00
    0    6/1/2014 0:00:00
    0    7/1/2014 0:03:00
    0    8/1/2014 0:03:00
    0    4/1/2014 0:11:00
    Name: Date/Time, dtype: object




```python
### converting 'Date/Time' feature into date-time..

final['Date/Time'] = pd.to_datetime(final['Date/Time'] , format="%m/%d/%Y %H:%M:%S")
```


```python
final['Date/Time'].dtype
```




    dtype('<M8[ns]')




```python
### extracting 'weekday' & 'hour' from 'Date/Time' feature..

final['day'] = final['Date/Time'].dt.day
final['hour'] = final['Date/Time'].dt.hour
```


```python
final.head(4)
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
      <th>Date/Time</th>
      <th>Lat</th>
      <th>Lon</th>
      <th>Base</th>
      <th>day</th>
      <th>hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014-09-01 00:01:00</td>
      <td>40.2201</td>
      <td>-74.0021</td>
      <td>B02512</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014-09-01 00:01:00</td>
      <td>40.7500</td>
      <td>-74.0027</td>
      <td>B02512</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014-09-01 00:03:00</td>
      <td>40.7559</td>
      <td>-73.9864</td>
      <td>B02512</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014-09-01 00:06:00</td>
      <td>40.7450</td>
      <td>-73.9889</td>
      <td>B02512</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
pivot = final.groupby(['day' , 'hour']).size().unstack()
```


```python
pivot

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
      <th>hour</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
    </tr>
    <tr>
      <th>day</th>
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
      <th>1</th>
      <td>3178</td>
      <td>1944</td>
      <td>1256</td>
      <td>1308</td>
      <td>1429</td>
      <td>2126</td>
      <td>3664</td>
      <td>5380</td>
      <td>5292</td>
      <td>4617</td>
      <td>...</td>
      <td>6933</td>
      <td>7910</td>
      <td>8633</td>
      <td>9511</td>
      <td>8604</td>
      <td>8001</td>
      <td>7315</td>
      <td>7803</td>
      <td>6268</td>
      <td>4050</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2435</td>
      <td>1569</td>
      <td>1087</td>
      <td>1414</td>
      <td>1876</td>
      <td>2812</td>
      <td>4920</td>
      <td>6544</td>
      <td>6310</td>
      <td>4712</td>
      <td>...</td>
      <td>6904</td>
      <td>8449</td>
      <td>10109</td>
      <td>11100</td>
      <td>11123</td>
      <td>9474</td>
      <td>8759</td>
      <td>8357</td>
      <td>6998</td>
      <td>5160</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3354</td>
      <td>2142</td>
      <td>1407</td>
      <td>1467</td>
      <td>1550</td>
      <td>2387</td>
      <td>4241</td>
      <td>5663</td>
      <td>5386</td>
      <td>4657</td>
      <td>...</td>
      <td>7226</td>
      <td>8850</td>
      <td>10314</td>
      <td>10491</td>
      <td>11239</td>
      <td>9599</td>
      <td>9026</td>
      <td>8531</td>
      <td>7142</td>
      <td>4686</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2897</td>
      <td>1688</td>
      <td>1199</td>
      <td>1424</td>
      <td>1696</td>
      <td>2581</td>
      <td>4592</td>
      <td>6029</td>
      <td>5704</td>
      <td>4744</td>
      <td>...</td>
      <td>7158</td>
      <td>8515</td>
      <td>9492</td>
      <td>10357</td>
      <td>10259</td>
      <td>9097</td>
      <td>8358</td>
      <td>8649</td>
      <td>7706</td>
      <td>5130</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2733</td>
      <td>1541</td>
      <td>1030</td>
      <td>1253</td>
      <td>1617</td>
      <td>2900</td>
      <td>4814</td>
      <td>6261</td>
      <td>6469</td>
      <td>5530</td>
      <td>...</td>
      <td>6955</td>
      <td>8312</td>
      <td>9609</td>
      <td>10699</td>
      <td>10170</td>
      <td>9430</td>
      <td>9354</td>
      <td>9610</td>
      <td>8853</td>
      <td>6518</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4537</td>
      <td>2864</td>
      <td>1864</td>
      <td>1555</td>
      <td>1551</td>
      <td>2162</td>
      <td>3642</td>
      <td>4766</td>
      <td>4942</td>
      <td>4401</td>
      <td>...</td>
      <td>7235</td>
      <td>8612</td>
      <td>9444</td>
      <td>9929</td>
      <td>9263</td>
      <td>8405</td>
      <td>8117</td>
      <td>8567</td>
      <td>7852</td>
      <td>5946</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3645</td>
      <td>2296</td>
      <td>1507</td>
      <td>1597</td>
      <td>1763</td>
      <td>2422</td>
      <td>4102</td>
      <td>5575</td>
      <td>5376</td>
      <td>4639</td>
      <td>...</td>
      <td>7276</td>
      <td>8474</td>
      <td>10393</td>
      <td>11013</td>
      <td>10573</td>
      <td>9472</td>
      <td>8691</td>
      <td>8525</td>
      <td>7194</td>
      <td>4801</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2830</td>
      <td>1646</td>
      <td>1123</td>
      <td>1483</td>
      <td>1889</td>
      <td>3224</td>
      <td>5431</td>
      <td>7361</td>
      <td>7357</td>
      <td>5703</td>
      <td>...</td>
      <td>7240</td>
      <td>8775</td>
      <td>9851</td>
      <td>10673</td>
      <td>9687</td>
      <td>8796</td>
      <td>8604</td>
      <td>8367</td>
      <td>6795</td>
      <td>4256</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2657</td>
      <td>1724</td>
      <td>1222</td>
      <td>1480</td>
      <td>1871</td>
      <td>3168</td>
      <td>5802</td>
      <td>7592</td>
      <td>7519</td>
      <td>5895</td>
      <td>...</td>
      <td>7877</td>
      <td>9220</td>
      <td>10270</td>
      <td>11910</td>
      <td>11449</td>
      <td>9804</td>
      <td>8909</td>
      <td>8665</td>
      <td>7499</td>
      <td>5203</td>
    </tr>
    <tr>
      <th>10</th>
      <td>3296</td>
      <td>2126</td>
      <td>1464</td>
      <td>1434</td>
      <td>1591</td>
      <td>2594</td>
      <td>4664</td>
      <td>6046</td>
      <td>6158</td>
      <td>5072</td>
      <td>...</td>
      <td>7612</td>
      <td>9578</td>
      <td>11045</td>
      <td>11875</td>
      <td>10934</td>
      <td>9613</td>
      <td>9687</td>
      <td>9240</td>
      <td>7766</td>
      <td>5496</td>
    </tr>
    <tr>
      <th>11</th>
      <td>3036</td>
      <td>1665</td>
      <td>1095</td>
      <td>1424</td>
      <td>1842</td>
      <td>2520</td>
      <td>4954</td>
      <td>6876</td>
      <td>6871</td>
      <td>5396</td>
      <td>...</td>
      <td>7503</td>
      <td>8920</td>
      <td>10125</td>
      <td>10898</td>
      <td>10361</td>
      <td>9327</td>
      <td>8824</td>
      <td>8730</td>
      <td>7771</td>
      <td>5360</td>
    </tr>
    <tr>
      <th>12</th>
      <td>3227</td>
      <td>2147</td>
      <td>1393</td>
      <td>1362</td>
      <td>1757</td>
      <td>2710</td>
      <td>4576</td>
      <td>6250</td>
      <td>6231</td>
      <td>5177</td>
      <td>...</td>
      <td>7743</td>
      <td>9390</td>
      <td>10734</td>
      <td>11713</td>
      <td>12216</td>
      <td>10393</td>
      <td>9965</td>
      <td>10310</td>
      <td>9992</td>
      <td>7945</td>
    </tr>
    <tr>
      <th>13</th>
      <td>5408</td>
      <td>3509</td>
      <td>2262</td>
      <td>1832</td>
      <td>1705</td>
      <td>2327</td>
      <td>4196</td>
      <td>5685</td>
      <td>6060</td>
      <td>5631</td>
      <td>...</td>
      <td>8200</td>
      <td>9264</td>
      <td>10534</td>
      <td>11826</td>
      <td>11450</td>
      <td>9921</td>
      <td>8705</td>
      <td>8423</td>
      <td>7363</td>
      <td>5936</td>
    </tr>
    <tr>
      <th>14</th>
      <td>3748</td>
      <td>2349</td>
      <td>1605</td>
      <td>1656</td>
      <td>1756</td>
      <td>2629</td>
      <td>4257</td>
      <td>5781</td>
      <td>5520</td>
      <td>4824</td>
      <td>...</td>
      <td>6963</td>
      <td>8192</td>
      <td>9511</td>
      <td>10115</td>
      <td>9553</td>
      <td>9146</td>
      <td>9182</td>
      <td>8589</td>
      <td>6891</td>
      <td>4460</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2497</td>
      <td>1515</td>
      <td>1087</td>
      <td>1381</td>
      <td>1862</td>
      <td>2980</td>
      <td>5050</td>
      <td>6837</td>
      <td>6729</td>
      <td>5201</td>
      <td>...</td>
      <td>7633</td>
      <td>8505</td>
      <td>10285</td>
      <td>11959</td>
      <td>11728</td>
      <td>11032</td>
      <td>10509</td>
      <td>9105</td>
      <td>7153</td>
      <td>4480</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2547</td>
      <td>1585</td>
      <td>1119</td>
      <td>1395</td>
      <td>1818</td>
      <td>2966</td>
      <td>5558</td>
      <td>7517</td>
      <td>7495</td>
      <td>5958</td>
      <td>...</td>
      <td>7597</td>
      <td>9290</td>
      <td>10804</td>
      <td>11773</td>
      <td>10855</td>
      <td>10924</td>
      <td>10142</td>
      <td>10374</td>
      <td>8094</td>
      <td>5380</td>
    </tr>
    <tr>
      <th>17</th>
      <td>3155</td>
      <td>2048</td>
      <td>1500</td>
      <td>1488</td>
      <td>1897</td>
      <td>2741</td>
      <td>4562</td>
      <td>6315</td>
      <td>5882</td>
      <td>4934</td>
      <td>...</td>
      <td>7472</td>
      <td>8997</td>
      <td>10323</td>
      <td>11236</td>
      <td>11089</td>
      <td>9919</td>
      <td>9935</td>
      <td>9823</td>
      <td>8362</td>
      <td>5699</td>
    </tr>
    <tr>
      <th>18</th>
      <td>3390</td>
      <td>2135</td>
      <td>1332</td>
      <td>1626</td>
      <td>1892</td>
      <td>2959</td>
      <td>4688</td>
      <td>6618</td>
      <td>6451</td>
      <td>5377</td>
      <td>...</td>
      <td>7534</td>
      <td>9040</td>
      <td>10274</td>
      <td>10692</td>
      <td>10338</td>
      <td>9551</td>
      <td>9310</td>
      <td>9285</td>
      <td>8015</td>
      <td>5492</td>
    </tr>
    <tr>
      <th>19</th>
      <td>3217</td>
      <td>2188</td>
      <td>1604</td>
      <td>1675</td>
      <td>1810</td>
      <td>2639</td>
      <td>4733</td>
      <td>6159</td>
      <td>6014</td>
      <td>5006</td>
      <td>...</td>
      <td>7374</td>
      <td>8898</td>
      <td>9893</td>
      <td>10741</td>
      <td>10429</td>
      <td>9701</td>
      <td>10051</td>
      <td>10049</td>
      <td>9090</td>
      <td>6666</td>
    </tr>
    <tr>
      <th>20</th>
      <td>4475</td>
      <td>3190</td>
      <td>2100</td>
      <td>1858</td>
      <td>1618</td>
      <td>2143</td>
      <td>3584</td>
      <td>4900</td>
      <td>5083</td>
      <td>4765</td>
      <td>...</td>
      <td>7462</td>
      <td>8630</td>
      <td>9448</td>
      <td>10046</td>
      <td>9272</td>
      <td>8592</td>
      <td>8614</td>
      <td>8703</td>
      <td>7787</td>
      <td>5907</td>
    </tr>
    <tr>
      <th>21</th>
      <td>4294</td>
      <td>3194</td>
      <td>1972</td>
      <td>1727</td>
      <td>1926</td>
      <td>2615</td>
      <td>4185</td>
      <td>5727</td>
      <td>5529</td>
      <td>4707</td>
      <td>...</td>
      <td>7064</td>
      <td>8127</td>
      <td>9483</td>
      <td>9817</td>
      <td>9291</td>
      <td>8317</td>
      <td>8107</td>
      <td>8245</td>
      <td>7362</td>
      <td>5231</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2787</td>
      <td>1637</td>
      <td>1175</td>
      <td>1468</td>
      <td>1934</td>
      <td>3151</td>
      <td>5204</td>
      <td>6872</td>
      <td>6850</td>
      <td>5198</td>
      <td>...</td>
      <td>7337</td>
      <td>9148</td>
      <td>10574</td>
      <td>10962</td>
      <td>9884</td>
      <td>8980</td>
      <td>8772</td>
      <td>8430</td>
      <td>6784</td>
      <td>4530</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2546</td>
      <td>1580</td>
      <td>1136</td>
      <td>1429</td>
      <td>1957</td>
      <td>3132</td>
      <td>5204</td>
      <td>6890</td>
      <td>6436</td>
      <td>5177</td>
      <td>...</td>
      <td>7575</td>
      <td>9309</td>
      <td>9980</td>
      <td>10341</td>
      <td>10823</td>
      <td>11347</td>
      <td>11447</td>
      <td>10347</td>
      <td>8637</td>
      <td>5577</td>
    </tr>
    <tr>
      <th>24</th>
      <td>3200</td>
      <td>2055</td>
      <td>1438</td>
      <td>1493</td>
      <td>1798</td>
      <td>2754</td>
      <td>4484</td>
      <td>6013</td>
      <td>5913</td>
      <td>5146</td>
      <td>...</td>
      <td>7083</td>
      <td>8706</td>
      <td>10366</td>
      <td>10786</td>
      <td>9772</td>
      <td>9080</td>
      <td>9213</td>
      <td>8831</td>
      <td>7480</td>
      <td>4456</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2405</td>
      <td>1499</td>
      <td>1072</td>
      <td>1439</td>
      <td>1943</td>
      <td>2973</td>
      <td>5356</td>
      <td>7627</td>
      <td>7078</td>
      <td>5994</td>
      <td>...</td>
      <td>7298</td>
      <td>8732</td>
      <td>9922</td>
      <td>10504</td>
      <td>10673</td>
      <td>9048</td>
      <td>8751</td>
      <td>9508</td>
      <td>8522</td>
      <td>6605</td>
    </tr>
    <tr>
      <th>26</th>
      <td>3810</td>
      <td>3065</td>
      <td>2046</td>
      <td>1806</td>
      <td>1730</td>
      <td>2337</td>
      <td>3776</td>
      <td>5172</td>
      <td>5071</td>
      <td>4808</td>
      <td>...</td>
      <td>7269</td>
      <td>8815</td>
      <td>9885</td>
      <td>10697</td>
      <td>10867</td>
      <td>10122</td>
      <td>9820</td>
      <td>10441</td>
      <td>9486</td>
      <td>7593</td>
    </tr>
    <tr>
      <th>27</th>
      <td>5196</td>
      <td>3635</td>
      <td>2352</td>
      <td>2055</td>
      <td>1723</td>
      <td>2336</td>
      <td>3539</td>
      <td>4937</td>
      <td>5053</td>
      <td>4771</td>
      <td>...</td>
      <td>7519</td>
      <td>8803</td>
      <td>9793</td>
      <td>9838</td>
      <td>9228</td>
      <td>8267</td>
      <td>7908</td>
      <td>8507</td>
      <td>7720</td>
      <td>6046</td>
    </tr>
    <tr>
      <th>28</th>
      <td>4123</td>
      <td>2646</td>
      <td>1843</td>
      <td>1802</td>
      <td>1883</td>
      <td>2793</td>
      <td>4290</td>
      <td>5715</td>
      <td>5671</td>
      <td>5206</td>
      <td>...</td>
      <td>7341</td>
      <td>8584</td>
      <td>9671</td>
      <td>9975</td>
      <td>9132</td>
      <td>8255</td>
      <td>8309</td>
      <td>7949</td>
      <td>6411</td>
      <td>4461</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2678</td>
      <td>1827</td>
      <td>1409</td>
      <td>1678</td>
      <td>1948</td>
      <td>3056</td>
      <td>5213</td>
      <td>6852</td>
      <td>6695</td>
      <td>5481</td>
      <td>...</td>
      <td>7630</td>
      <td>9249</td>
      <td>10105</td>
      <td>11113</td>
      <td>10411</td>
      <td>9301</td>
      <td>9270</td>
      <td>9114</td>
      <td>6992</td>
      <td>4323</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2401</td>
      <td>1510</td>
      <td>1112</td>
      <td>1403</td>
      <td>1841</td>
      <td>3216</td>
      <td>5757</td>
      <td>7596</td>
      <td>7611</td>
      <td>6064</td>
      <td>...</td>
      <td>8396</td>
      <td>10243</td>
      <td>11554</td>
      <td>12126</td>
      <td>12561</td>
      <td>11024</td>
      <td>10836</td>
      <td>10042</td>
      <td>8275</td>
      <td>4723</td>
    </tr>
    <tr>
      <th>31</th>
      <td>2174</td>
      <td>1394</td>
      <td>1087</td>
      <td>919</td>
      <td>773</td>
      <td>997</td>
      <td>1561</td>
      <td>2169</td>
      <td>2410</td>
      <td>2525</td>
      <td>...</td>
      <td>4104</td>
      <td>5099</td>
      <td>5386</td>
      <td>5308</td>
      <td>5350</td>
      <td>4898</td>
      <td>4819</td>
      <td>5064</td>
      <td>5164</td>
      <td>3961</td>
    </tr>
  </tbody>
</table>
<p>31 rows × 24 columns</p>
</div>




```python
### styling dataframe
pivot.style.background_gradient()
```




<style type="text/css">
#T_ee5c2_row0_col0, #T_ee5c2_row8_col23, #T_ee5c2_row23_col2 {
  background-color: #bcc7e1;
  color: #000000;
}
#T_ee5c2_row0_col1 {
  background-color: #d2d2e7;
  color: #000000;
}
#T_ee5c2_row0_col2 {
  background-color: #e2dfee;
  color: #000000;
}
#T_ee5c2_row0_col3 {
  background-color: #b1c2de;
  color: #000000;
}
#T_ee5c2_row0_col4, #T_ee5c2_row0_col8 {
  background-color: #5a9ec9;
  color: #f1f1f1;
}
#T_ee5c2_row0_col5, #T_ee5c2_row23_col3, #T_ee5c2_row25_col0, #T_ee5c2_row26_col7 {
  background-color: #71a8ce;
  color: #f1f1f1;
}
#T_ee5c2_row0_col6, #T_ee5c2_row8_col3, #T_ee5c2_row12_col23, #T_ee5c2_row20_col20 {
  background-color: #76aad0;
  color: #f1f1f1;
}
#T_ee5c2_row0_col7, #T_ee5c2_row0_col11, #T_ee5c2_row12_col20, #T_ee5c2_row20_col15 {
  background-color: #4897c4;
  color: #f1f1f1;
}
#T_ee5c2_row0_col9, #T_ee5c2_row17_col22, #T_ee5c2_row20_col21, #T_ee5c2_row24_col20 {
  background-color: #4697c4;
  color: #f1f1f1;
}
#T_ee5c2_row0_col10, #T_ee5c2_row1_col20, #T_ee5c2_row6_col3, #T_ee5c2_row6_col9, #T_ee5c2_row12_col5, #T_ee5c2_row21_col20 {
  background-color: #4496c3;
  color: #f1f1f1;
}
#T_ee5c2_row0_col12, #T_ee5c2_row6_col8 {
  background-color: #529bc7;
  color: #f1f1f1;
}
#T_ee5c2_row0_col13, #T_ee5c2_row2_col20, #T_ee5c2_row3_col8, #T_ee5c2_row3_col10, #T_ee5c2_row13_col6, #T_ee5c2_row13_col12, #T_ee5c2_row19_col9, #T_ee5c2_row21_col19, #T_ee5c2_row26_col9 {
  background-color: #328dbf;
  color: #f1f1f1;
}
#T_ee5c2_row0_col14, #T_ee5c2_row2_col4, #T_ee5c2_row5_col4, #T_ee5c2_row5_col16, #T_ee5c2_row13_col19, #T_ee5c2_row13_col20, #T_ee5c2_row19_col16 {
  background-color: #2987bc;
  color: #f1f1f1;
}
#T_ee5c2_row0_col15, #T_ee5c2_row5_col19, #T_ee5c2_row19_col18, #T_ee5c2_row19_col22, #T_ee5c2_row20_col18 {
  background-color: #5ea0ca;
  color: #f1f1f1;
}
#T_ee5c2_row0_col16, #T_ee5c2_row3_col22, #T_ee5c2_row27_col18, #T_ee5c2_row27_col20 {
  background-color: #67a4cc;
  color: #f1f1f1;
}
#T_ee5c2_row0_col17, #T_ee5c2_row7_col21, #T_ee5c2_row8_col20, #T_ee5c2_row20_col9, #T_ee5c2_row27_col2 {
  background-color: #3b92c1;
  color: #f1f1f1;
}
#T_ee5c2_row0_col18 {
  background-color: #88b1d4;
  color: #000000;
}
#T_ee5c2_row0_col19, #T_ee5c2_row2_col3, #T_ee5c2_row8_col22, #T_ee5c2_row21_col3 {
  background-color: #7bacd1;
  color: #f1f1f1;
}
#T_ee5c2_row0_col20, #T_ee5c2_row17_col0, #T_ee5c2_row28_col22 {
  background-color: #a5bddb;
  color: #000000;
}
#T_ee5c2_row0_col21, #T_ee5c2_row25_col8, #T_ee5c2_row26_col8 {
  background-color: #6fa7ce;
  color: #f1f1f1;
}
#T_ee5c2_row0_col22, #T_ee5c2_row17_col2 {
  background-color: #d5d5e8;
  color: #000000;
}
#T_ee5c2_row0_col23 {
  background-color: #fcf4fa;
  color: #000000;
}
#T_ee5c2_row1_col0, #T_ee5c2_row22_col2 {
  background-color: #f3edf5;
  color: #000000;
}
#T_ee5c2_row1_col1 {
  background-color: #f4edf6;
  color: #000000;
}
#T_ee5c2_row1_col2, #T_ee5c2_row10_col2, #T_ee5c2_row14_col2, #T_ee5c2_row24_col1, #T_ee5c2_row30_col2 {
  background-color: #f8f1f8;
  color: #000000;
}
#T_ee5c2_row1_col3, #T_ee5c2_row13_col2, #T_ee5c2_row16_col23, #T_ee5c2_row18_col2 {
  background-color: #8eb3d5;
  color: #000000;
}
#T_ee5c2_row1_col4, #T_ee5c2_row12_col2 {
  background-color: #034a74;
  color: #f1f1f1;
}
#T_ee5c2_row1_col5, #T_ee5c2_row14_col10, #T_ee5c2_row15_col11, #T_ee5c2_row15_col14, #T_ee5c2_row15_col15, #T_ee5c2_row16_col13, #T_ee5c2_row17_col7, #T_ee5c2_row18_col22 {
  background-color: #04649e;
  color: #f1f1f1;
}
#T_ee5c2_row1_col6, #T_ee5c2_row4_col17, #T_ee5c2_row8_col13, #T_ee5c2_row8_col16, #T_ee5c2_row10_col14, #T_ee5c2_row17_col16, #T_ee5c2_row17_col17, #T_ee5c2_row18_col20, #T_ee5c2_row21_col10, #T_ee5c2_row24_col12, #T_ee5c2_row25_col17 {
  background-color: #0569a4;
  color: #f1f1f1;
}
#T_ee5c2_row1_col7, #T_ee5c2_row8_col15, #T_ee5c2_row12_col3, #T_ee5c2_row15_col20, #T_ee5c2_row19_col1, #T_ee5c2_row19_col12, #T_ee5c2_row20_col1, #T_ee5c2_row23_col17, #T_ee5c2_row26_col4 {
  background-color: #0567a1;
  color: #f1f1f1;
}
#T_ee5c2_row1_col8, #T_ee5c2_row4_col7, #T_ee5c2_row11_col7, #T_ee5c2_row11_col9, #T_ee5c2_row18_col6, #T_ee5c2_row22_col9 {
  background-color: #0570b0;
  color: #f1f1f1;
}
#T_ee5c2_row1_col9, #T_ee5c2_row20_col6 {
  background-color: #3991c1;
  color: #f1f1f1;
}
#T_ee5c2_row1_col10, #T_ee5c2_row1_col14, #T_ee5c2_row5_col1, #T_ee5c2_row5_col10, #T_ee5c2_row6_col15, #T_ee5c2_row13_col21, #T_ee5c2_row20_col0, #T_ee5c2_row23_col12, #T_ee5c2_row28_col12 {
  background-color: #2a88bc;
  color: #f1f1f1;
}
#T_ee5c2_row1_col11, #T_ee5c2_row3_col11, #T_ee5c2_row13_col7, #T_ee5c2_row14_col15, #T_ee5c2_row16_col22, #T_ee5c2_row20_col17, #T_ee5c2_row23_col20, #T_ee5c2_row24_col23 {
  background-color: #2786bb;
  color: #f1f1f1;
}
#T_ee5c2_row1_col12, #T_ee5c2_row4_col23, #T_ee5c2_row6_col21, #T_ee5c2_row12_col7, #T_ee5c2_row24_col19, #T_ee5c2_row27_col6, #T_ee5c2_row29_col22 {
  background-color: #2f8bbe;
  color: #f1f1f1;
}
#T_ee5c2_row1_col13, #T_ee5c2_row4_col13, #T_ee5c2_row15_col22 {
  background-color: #3f93c2;
  color: #f1f1f1;
}
#T_ee5c2_row1_col15, #T_ee5c2_row2_col10, #T_ee5c2_row3_col19, #T_ee5c2_row5_col21, #T_ee5c2_row13_col3, #T_ee5c2_row13_col9, #T_ee5c2_row20_col7, #T_ee5c2_row23_col19, #T_ee5c2_row27_col7 {
  background-color: #2c89bd;
  color: #f1f1f1;
}
#T_ee5c2_row1_col16, #T_ee5c2_row4_col6, #T_ee5c2_row10_col16, #T_ee5c2_row11_col5, #T_ee5c2_row11_col11, #T_ee5c2_row17_col15, #T_ee5c2_row25_col2, #T_ee5c2_row26_col10 {
  background-color: #056dab;
  color: #f1f1f1;
}
#T_ee5c2_row1_col17, #T_ee5c2_row4_col9, #T_ee5c2_row11_col14, #T_ee5c2_row19_col13, #T_ee5c2_row28_col17 {
  background-color: #045e94;
  color: #f1f1f1;
}
#T_ee5c2_row1_col18, #T_ee5c2_row2_col16, #T_ee5c2_row10_col6, #T_ee5c2_row10_col11, #T_ee5c2_row11_col13, #T_ee5c2_row16_col16, #T_ee5c2_row17_col14 {
  background-color: #0567a2;
  color: #f1f1f1;
}
#T_ee5c2_row1_col19, #T_ee5c2_row2_col13, #T_ee5c2_row3_col7, #T_ee5c2_row6_col19, #T_ee5c2_row9_col7, #T_ee5c2_row11_col6, #T_ee5c2_row16_col6 {
  background-color: #157ab5;
  color: #f1f1f1;
}
#T_ee5c2_row1_col21, #T_ee5c2_row23_col18 {
  background-color: #3d93c2;
  color: #f1f1f1;
}
#T_ee5c2_row1_col22 {
  background-color: #a4bcda;
  color: #000000;
}
#T_ee5c2_row1_col23, #T_ee5c2_row16_col0 {
  background-color: #bfc9e1;
  color: #000000;
}
#T_ee5c2_row2_col0 {
  background-color: #a9bfdc;
  color: #000000;
}
#T_ee5c2_row2_col1, #T_ee5c2_row21_col22 {
  background-color: #b4c4df;
  color: #000000;
}
#T_ee5c2_row2_col2, #T_ee5c2_row28_col2 {
  background-color: #c4cbe3;
  color: #000000;
}
#T_ee5c2_row2_col5, #T_ee5c2_row4_col15, #T_ee5c2_row6_col7, #T_ee5c2_row12_col6, #T_ee5c2_row12_col21, #T_ee5c2_row13_col13, #T_ee5c2_row17_col3 {
  background-color: #3790c0;
  color: #f1f1f1;
}
#T_ee5c2_row2_col6, #T_ee5c2_row5_col2 {
  background-color: #348ebf;
  color: #f1f1f1;
}
#T_ee5c2_row2_col7, #T_ee5c2_row6_col5, #T_ee5c2_row26_col21 {
  background-color: #308cbe;
  color: #f1f1f1;
}
#T_ee5c2_row2_col8, #T_ee5c2_row7_col20, #T_ee5c2_row19_col19, #T_ee5c2_row19_col20 {
  background-color: #509ac6;
  color: #f1f1f1;
}
#T_ee5c2_row2_col9, #T_ee5c2_row4_col12, #T_ee5c2_row7_col19, #T_ee5c2_row10_col20, #T_ee5c2_row25_col5, #T_ee5c2_row27_col0 {
  background-color: #4094c3;
  color: #f1f1f1;
}
#T_ee5c2_row2_col11, #T_ee5c2_row9_col4, #T_ee5c2_row20_col14, #T_ee5c2_row23_col6 {
  background-color: #1c7fb8;
  color: #f1f1f1;
}
#T_ee5c2_row2_col12, #T_ee5c2_row4_col19, #T_ee5c2_row12_col8, #T_ee5c2_row18_col9, #T_ee5c2_row23_col15, #T_ee5c2_row23_col21, #T_ee5c2_row25_col12, #T_ee5c2_row28_col18 {
  background-color: #187cb6;
  color: #f1f1f1;
}
#T_ee5c2_row2_col14, #T_ee5c2_row2_col15, #T_ee5c2_row2_col19, #T_ee5c2_row3_col13, #T_ee5c2_row5_col14, #T_ee5c2_row7_col12, #T_ee5c2_row25_col10, #T_ee5c2_row25_col13, #T_ee5c2_row25_col16, #T_ee5c2_row27_col12 {
  background-color: #0d75b3;
  color: #f1f1f1;
}
#T_ee5c2_row2_col17, #T_ee5c2_row8_col19, #T_ee5c2_row16_col7, #T_ee5c2_row18_col12, #T_ee5c2_row22_col18 {
  background-color: #056ead;
  color: #f1f1f1;
}
#T_ee5c2_row2_col18, #T_ee5c2_row9_col13, #T_ee5c2_row9_col14, #T_ee5c2_row10_col17, #T_ee5c2_row17_col11, #T_ee5c2_row22_col15 {
  background-color: #04649d;
  color: #f1f1f1;
}
#T_ee5c2_row2_col21, #T_ee5c2_row3_col12, #T_ee5c2_row25_col9 {
  background-color: #2d8abd;
  color: #f1f1f1;
}
#T_ee5c2_row2_col22, #T_ee5c2_row14_col3 {
  background-color: #99b8d8;
  color: #000000;
}
#T_ee5c2_row2_col23 {
  background-color: #e0dded;
  color: #000000;
}
#T_ee5c2_row3_col0 {
  background-color: #d6d6e9;
  color: #000000;
}
#T_ee5c2_row3_col1, #T_ee5c2_row14_col23 {
  background-color: #ebe6f2;
  color: #000000;
}
#T_ee5c2_row3_col2, #T_ee5c2_row13_col23, #T_ee5c2_row27_col23 {
  background-color: #ece7f2;
  color: #000000;
}
#T_ee5c2_row3_col3, #T_ee5c2_row10_col3 {
  background-color: #8bb2d4;
  color: #000000;
}
#T_ee5c2_row3_col4, #T_ee5c2_row4_col8, #T_ee5c2_row12_col19, #T_ee5c2_row16_col19, #T_ee5c2_row18_col13, #T_ee5c2_row25_col3, #T_ee5c2_row28_col10 {
  background-color: #056ba7;
  color: #f1f1f1;
}
#T_ee5c2_row3_col5, #T_ee5c2_row3_col6, #T_ee5c2_row3_col14, #T_ee5c2_row4_col4, #T_ee5c2_row5_col12, #T_ee5c2_row7_col15, #T_ee5c2_row16_col10, #T_ee5c2_row19_col0, #T_ee5c2_row19_col4, #T_ee5c2_row20_col2, #T_ee5c2_row20_col3, #T_ee5c2_row26_col16 {
  background-color: #1379b5;
  color: #f1f1f1;
}
#T_ee5c2_row3_col9, #T_ee5c2_row21_col18, #T_ee5c2_row21_col21, #T_ee5c2_row27_col8 {
  background-color: #358fc0;
  color: #f1f1f1;
}
#T_ee5c2_row3_col15, #T_ee5c2_row3_col16, #T_ee5c2_row3_col21, #T_ee5c2_row4_col14, #T_ee5c2_row13_col14, #T_ee5c2_row16_col8, #T_ee5c2_row18_col3, #T_ee5c2_row20_col16, #T_ee5c2_row26_col17 {
  background-color: #2685bb;
  color: #f1f1f1;
}
#T_ee5c2_row3_col17, #T_ee5c2_row6_col13, #T_ee5c2_row6_col14, #T_ee5c2_row15_col12, #T_ee5c2_row18_col10, #T_ee5c2_row18_col15, #T_ee5c2_row23_col9 {
  background-color: #0872b1;
  color: #f1f1f1;
}
#T_ee5c2_row3_col18, #T_ee5c2_row5_col15, #T_ee5c2_row10_col21, #T_ee5c2_row16_col9, #T_ee5c2_row28_col19 {
  background-color: #2081b9;
  color: #f1f1f1;
}
#T_ee5c2_row3_col20 {
  background-color: #63a2cb;
  color: #f1f1f1;
}
#T_ee5c2_row3_col23, #T_ee5c2_row4_col3, #T_ee5c2_row23_col1 {
  background-color: #c1cae2;
  color: #000000;
}
#T_ee5c2_row4_col0 {
  background-color: #e1dfed;
  color: #000000;
}
#T_ee5c2_row4_col1 {
  background-color: #f5eff6;
  color: #000000;
}
#T_ee5c2_row4_col2, #T_ee5c2_row30_col0, #T_ee5c2_row30_col1, #T_ee5c2_row30_col3, #T_ee5c2_row30_col4, #T_ee5c2_row30_col5, #T_ee5c2_row30_col6, #T_ee5c2_row30_col7, #T_ee5c2_row30_col8, #T_ee5c2_row30_col9, #T_ee5c2_row30_col10, #T_ee5c2_row30_col11, #T_ee5c2_row30_col12, #T_ee5c2_row30_col13, #T_ee5c2_row30_col14, #T_ee5c2_row30_col15, #T_ee5c2_row30_col16, #T_ee5c2_row30_col17, #T_ee5c2_row30_col18, #T_ee5c2_row30_col19, #T_ee5c2_row30_col20, #T_ee5c2_row30_col21, #T_ee5c2_row30_col22, #T_ee5c2_row30_col23 {
  background-color: #fff7fb;
  color: #000000;
}
#T_ee5c2_row4_col5, #T_ee5c2_row11_col19, #T_ee5c2_row12_col12, #T_ee5c2_row14_col7, #T_ee5c2_row21_col8 {
  background-color: #045e93;
  color: #f1f1f1;
}
#T_ee5c2_row4_col10, #T_ee5c2_row5_col13, #T_ee5c2_row14_col21, #T_ee5c2_row19_col10, #T_ee5c2_row21_col14, #T_ee5c2_row28_col21 {
  background-color: #056faf;
  color: #f1f1f1;
}
#T_ee5c2_row4_col11, #T_ee5c2_row23_col8 {
  background-color: #2383ba;
  color: #f1f1f1;
}
#T_ee5c2_row4_col16, #T_ee5c2_row4_col20, #T_ee5c2_row6_col10, #T_ee5c2_row10_col5, #T_ee5c2_row10_col19, #T_ee5c2_row13_col10, #T_ee5c2_row19_col15, #T_ee5c2_row20_col10, #T_ee5c2_row27_col17 {
  background-color: #1e80b8;
  color: #f1f1f1;
}
#T_ee5c2_row4_col18, #T_ee5c2_row8_col21, #T_ee5c2_row13_col16, #T_ee5c2_row28_col3, #T_ee5c2_row28_col20 {
  background-color: #2484ba;
  color: #f1f1f1;
}
#T_ee5c2_row4_col21, #T_ee5c2_row8_col18, #T_ee5c2_row12_col18 {
  background-color: #045f95;
  color: #f1f1f1;
}
#T_ee5c2_row4_col22, #T_ee5c2_row7_col13, #T_ee5c2_row15_col18, #T_ee5c2_row16_col11, #T_ee5c2_row18_col14, #T_ee5c2_row22_col11, #T_ee5c2_row23_col11, #T_ee5c2_row24_col17, #T_ee5c2_row25_col18, #T_ee5c2_row28_col13, #T_ee5c2_row28_col16 {
  background-color: #056dac;
  color: #f1f1f1;
}
#T_ee5c2_row5_col0, #T_ee5c2_row7_col14, #T_ee5c2_row8_col12, #T_ee5c2_row9_col6, #T_ee5c2_row9_col19, #T_ee5c2_row13_col5, #T_ee5c2_row17_col12, #T_ee5c2_row18_col7, #T_ee5c2_row18_col16, #T_ee5c2_row22_col10 {
  background-color: #0c74b2;
  color: #f1f1f1;
}
#T_ee5c2_row5_col3, #T_ee5c2_row27_col1 {
  background-color: #569dc8;
  color: #f1f1f1;
}
#T_ee5c2_row5_col5, #T_ee5c2_row25_col6, #T_ee5c2_row26_col19, #T_ee5c2_row26_col23, #T_ee5c2_row27_col19 {
  background-color: #69a5cc;
  color: #f1f1f1;
}
#T_ee5c2_row5_col6, #T_ee5c2_row19_col23 {
  background-color: #78abd0;
  color: #f1f1f1;
}
#T_ee5c2_row5_col7 {
  background-color: #7eadd1;
  color: #f1f1f1;
}
#T_ee5c2_row5_col8, #T_ee5c2_row13_col0 {
  background-color: #79abd0;
  color: #f1f1f1;
}
#T_ee5c2_row5_col9, #T_ee5c2_row20_col19, #T_ee5c2_row26_col22 {
  background-color: #65a3cb;
  color: #f1f1f1;
}
#T_ee5c2_row5_col11, #T_ee5c2_row6_col18, #T_ee5c2_row7_col16, #T_ee5c2_row20_col5, #T_ee5c2_row20_col12, #T_ee5c2_row25_col11 {
  background-color: #0f76b3;
  color: #f1f1f1;
}
#T_ee5c2_row5_col17, #T_ee5c2_row17_col20, #T_ee5c2_row18_col23, #T_ee5c2_row19_col21, #T_ee5c2_row23_col13, #T_ee5c2_row27_col15 {
  background-color: #2182b9;
  color: #f1f1f1;
}
#T_ee5c2_row5_col18, #T_ee5c2_row10_col22 {
  background-color: #60a1ca;
  color: #f1f1f1;
}
#T_ee5c2_row5_col20, #T_ee5c2_row5_col23, #T_ee5c2_row7_col3 {
  background-color: #75a9cf;
  color: #f1f1f1;
}
#T_ee5c2_row5_col22 {
  background-color: #589ec8;
  color: #f1f1f1;
}
#T_ee5c2_row6_col0, #T_ee5c2_row9_col3, #T_ee5c2_row12_col22, #T_ee5c2_row20_col22 {
  background-color: #86b0d3;
  color: #000000;
}
#T_ee5c2_row6_col1, #T_ee5c2_row22_col23 {
  background-color: #9ab8d8;
  color: #000000;
}
#T_ee5c2_row6_col2 {
  background-color: #abbfdc;
  color: #000000;
}
#T_ee5c2_row6_col4, #T_ee5c2_row6_col17, #T_ee5c2_row24_col10 {
  background-color: #046097;
  color: #f1f1f1;
}
#T_ee5c2_row6_col6, #T_ee5c2_row7_col18, #T_ee5c2_row13_col8, #T_ee5c2_row13_col15, #T_ee5c2_row20_col8, #T_ee5c2_row26_col5 {
  background-color: #4295c3;
  color: #f1f1f1;
}
#T_ee5c2_row6_col11, #T_ee5c2_row9_col8, #T_ee5c2_row9_col9, #T_ee5c2_row17_col19, #T_ee5c2_row22_col22, #T_ee5c2_row25_col15, #T_ee5c2_row26_col15, #T_ee5c2_row28_col11 {
  background-color: #1077b4;
  color: #f1f1f1;
}
#T_ee5c2_row6_col12, #T_ee5c2_row10_col18, #T_ee5c2_row17_col18, #T_ee5c2_row18_col8, #T_ee5c2_row19_col17, #T_ee5c2_row23_col14, #T_ee5c2_row27_col16 {
  background-color: #1b7eb7;
  color: #f1f1f1;
}
#T_ee5c2_row6_col16, #T_ee5c2_row10_col9, #T_ee5c2_row12_col15, #T_ee5c2_row19_col2, #T_ee5c2_row22_col14, #T_ee5c2_row25_col19 {
  background-color: #05659f;
  color: #f1f1f1;
}
#T_ee5c2_row6_col20, #T_ee5c2_row13_col18 {
  background-color: #4a98c5;
  color: #f1f1f1;
}
#T_ee5c2_row6_col22, #T_ee5c2_row15_col3 {
  background-color: #94b6d7;
  color: #000000;
}
#T_ee5c2_row6_col23 {
  background-color: #d9d8ea;
  color: #000000;
}
#T_ee5c2_row7_col0 {
  background-color: #dbdaeb;
  color: #000000;
}
#T_ee5c2_row7_col1, #T_ee5c2_row21_col2 {
  background-color: #eee9f3;
  color: #000000;
}
#T_ee5c2_row7_col2, #T_ee5c2_row7_col23, #T_ee5c2_row24_col0 {
  background-color: #f4eef6;
  color: #000000;
}
#T_ee5c2_row7_col4, #T_ee5c2_row12_col1, #T_ee5c2_row15_col6, #T_ee5c2_row17_col4 {
  background-color: #03476f;
  color: #f1f1f1;
}
#T_ee5c2_row7_col5, #T_ee5c2_row8_col6, #T_ee5c2_row11_col22, #T_ee5c2_row11_col23, #T_ee5c2_row12_col0, #T_ee5c2_row22_col4, #T_ee5c2_row22_col19, #T_ee5c2_row22_col20, #T_ee5c2_row24_col7, #T_ee5c2_row25_col21, #T_ee5c2_row26_col1, #T_ee5c2_row26_col2, #T_ee5c2_row26_col3, #T_ee5c2_row29_col5, #T_ee5c2_row29_col8, #T_ee5c2_row29_col9, #T_ee5c2_row29_col10, #T_ee5c2_row29_col11, #T_ee5c2_row29_col12, #T_ee5c2_row29_col13, #T_ee5c2_row29_col14, #T_ee5c2_row29_col15, #T_ee5c2_row29_col16, #T_ee5c2_row29_col17, #T_ee5c2_row29_col18 {
  background-color: #023858;
  color: #f1f1f1;
}
#T_ee5c2_row7_col6, #T_ee5c2_row25_col23 {
  background-color: #034f7d;
  color: #f1f1f1;
}
#T_ee5c2_row7_col7, #T_ee5c2_row7_col8, #T_ee5c2_row8_col9, #T_ee5c2_row11_col18, #T_ee5c2_row14_col19, #T_ee5c2_row16_col4, #T_ee5c2_row29_col19 {
  background-color: #03456c;
  color: #f1f1f1;
}
#T_ee5c2_row7_col9, #T_ee5c2_row15_col10, #T_ee5c2_row24_col6, #T_ee5c2_row24_col8, #T_ee5c2_row25_col22 {
  background-color: #045483;
  color: #f1f1f1;
}
#T_ee5c2_row7_col10, #T_ee5c2_row9_col11, #T_ee5c2_row14_col16, #T_ee5c2_row16_col18, #T_ee5c2_row18_col17, #T_ee5c2_row26_col14 {
  background-color: #0568a3;
  color: #f1f1f1;
}
#T_ee5c2_row7_col11, #T_ee5c2_row9_col18, #T_ee5c2_row9_col21, #T_ee5c2_row10_col10, #T_ee5c2_row11_col20, #T_ee5c2_row17_col8, #T_ee5c2_row21_col11, #T_ee5c2_row22_col8, #T_ee5c2_row27_col3 {
  background-color: #056ba9;
  color: #f1f1f1;
}
#T_ee5c2_row7_col17, #T_ee5c2_row12_col4, #T_ee5c2_row21_col15, #T_ee5c2_row23_col5 {
  background-color: #0569a5;
  color: #f1f1f1;
}
#T_ee5c2_row7_col22, #T_ee5c2_row11_col1 {
  background-color: #b3c3de;
  color: #000000;
}
#T_ee5c2_row8_col0, #T_ee5c2_row8_col1, #T_ee5c2_row8_col2 {
  background-color: #e7e3f0;
  color: #000000;
}
#T_ee5c2_row8_col4, #T_ee5c2_row18_col21, #T_ee5c2_row29_col21 {
  background-color: #034b76;
  color: #f1f1f1;
}
#T_ee5c2_row8_col5, #T_ee5c2_row11_col21, #T_ee5c2_row14_col17, #T_ee5c2_row20_col4 {
  background-color: #023e62;
  color: #f1f1f1;
}
#T_ee5c2_row8_col7, #T_ee5c2_row28_col4, #T_ee5c2_row29_col7 {
  background-color: #02395a;
  color: #f1f1f1;
}
#T_ee5c2_row8_col8, #T_ee5c2_row21_col4, #T_ee5c2_row22_col21 {
  background-color: #023c5f;
  color: #f1f1f1;
}
#T_ee5c2_row8_col10, #T_ee5c2_row11_col4, #T_ee5c2_row13_col4, #T_ee5c2_row14_col8, #T_ee5c2_row21_col17, #T_ee5c2_row26_col12 {
  background-color: #046299;
  color: #f1f1f1;
}
#T_ee5c2_row8_col11, #T_ee5c2_row10_col13, #T_ee5c2_row17_col9, #T_ee5c2_row23_col16, #T_ee5c2_row25_col4, #T_ee5c2_row27_col5, #T_ee5c2_row28_col15 {
  background-color: #0566a0;
  color: #f1f1f1;
}
#T_ee5c2_row8_col14, #T_ee5c2_row15_col4, #T_ee5c2_row17_col5 {
  background-color: #04588a;
  color: #f1f1f1;
}
#T_ee5c2_row8_col17, #T_ee5c2_row21_col5 {
  background-color: #034165;
  color: #f1f1f1;
}
#T_ee5c2_row9_col0 {
  background-color: #b0c2de;
  color: #000000;
}
#T_ee5c2_row9_col1, #T_ee5c2_row11_col0 {
  background-color: #b7c5df;
  color: #000000;
}
#T_ee5c2_row9_col2, #T_ee5c2_row17_col1 {
  background-color: #b5c4df;
  color: #000000;
}
#T_ee5c2_row9_col5, #T_ee5c2_row27_col13 {
  background-color: #1278b4;
  color: #f1f1f1;
}
#T_ee5c2_row9_col10, #T_ee5c2_row13_col11, #T_ee5c2_row13_col17, #T_ee5c2_row18_col18, #T_ee5c2_row20_col13, #T_ee5c2_row23_col7, #T_ee5c2_row24_col15 {
  background-color: #167bb6;
  color: #f1f1f1;
}
#T_ee5c2_row9_col12, #T_ee5c2_row9_col20, #T_ee5c2_row10_col12, #T_ee5c2_row11_col8, #T_ee5c2_row14_col12, #T_ee5c2_row15_col13, #T_ee5c2_row17_col6, #T_ee5c2_row18_col5, #T_ee5c2_row20_col11, #T_ee5c2_row21_col12, #T_ee5c2_row22_col12, #T_ee5c2_row22_col17, #T_ee5c2_row24_col13, #T_ee5c2_row24_col16, #T_ee5c2_row24_col18, #T_ee5c2_row25_col14 {
  background-color: #0a73b2;
  color: #f1f1f1;
}
#T_ee5c2_row9_col15, #T_ee5c2_row16_col17, #T_ee5c2_row19_col11 {
  background-color: #045b8e;
  color: #f1f1f1;
}
#T_ee5c2_row9_col16 {
  background-color: #034e7b;
  color: #f1f1f1;
}
#T_ee5c2_row9_col17 {
  background-color: #034267;
  color: #f1f1f1;
}
#T_ee5c2_row9_col22, #T_ee5c2_row26_col18, #T_ee5c2_row27_col21 {
  background-color: #62a2cb;
  color: #f1f1f1;
}
#T_ee5c2_row9_col23, #T_ee5c2_row17_col23 {
  background-color: #a2bcda;
  color: #000000;
}
#T_ee5c2_row10_col0 {
  background-color: #cacee5;
  color: #000000;
}
#T_ee5c2_row10_col1 {
  background-color: #ede8f3;
  color: #000000;
}
#T_ee5c2_row10_col4 {
  background-color: #045280;
  color: #f1f1f1;
}
#T_ee5c2_row10_col7, #T_ee5c2_row21_col7, #T_ee5c2_row28_col6 {
  background-color: #045c90;
  color: #f1f1f1;
}
#T_ee5c2_row10_col8, #T_ee5c2_row12_col13, #T_ee5c2_row14_col20, #T_ee5c2_row21_col6, #T_ee5c2_row22_col6, #T_ee5c2_row28_col7 {
  background-color: #045d92;
  color: #f1f1f1;
}
#T_ee5c2_row10_col15, #T_ee5c2_row18_col11, #T_ee5c2_row18_col19, #T_ee5c2_row22_col13, #T_ee5c2_row22_col16, #T_ee5c2_row24_col14, #T_ee5c2_row25_col1 {
  background-color: #0771b1;
  color: #f1f1f1;
}
#T_ee5c2_row10_col23 {
  background-color: #afc1dd;
  color: #000000;
}
#T_ee5c2_row11_col2 {
  background-color: #c8cde4;
  color: #000000;
}
#T_ee5c2_row11_col3 {
  background-color: #a1bbda;
  color: #000000;
}
#T_ee5c2_row11_col10, #T_ee5c2_row11_col12, #T_ee5c2_row14_col9, #T_ee5c2_row16_col15, #T_ee5c2_row17_col10, #T_ee5c2_row21_col9, #T_ee5c2_row25_col20, #T_ee5c2_row27_col9, #T_ee5c2_row27_col14 {
  background-color: #056fae;
  color: #f1f1f1;
}
#T_ee5c2_row11_col15, #T_ee5c2_row12_col16, #T_ee5c2_row28_col9 {
  background-color: #046198;
  color: #f1f1f1;
}
#T_ee5c2_row11_col16, #T_ee5c2_row22_col7, #T_ee5c2_row23_col4 {
  background-color: #045b8f;
  color: #f1f1f1;
}
#T_ee5c2_row11_col17, #T_ee5c2_row27_col4 {
  background-color: #034871;
  color: #f1f1f1;
}
#T_ee5c2_row12_col9, #T_ee5c2_row15_col16, #T_ee5c2_row18_col4 {
  background-color: #04598c;
  color: #f1f1f1;
}
#T_ee5c2_row12_col10, #T_ee5c2_row21_col16 {
  background-color: #046096;
  color: #f1f1f1;
}
#T_ee5c2_row12_col11, #T_ee5c2_row14_col5, #T_ee5c2_row24_col5 {
  background-color: #045687;
  color: #f1f1f1;
}
#T_ee5c2_row12_col14, #T_ee5c2_row12_col17 {
  background-color: #03446a;
  color: #f1f1f1;
}
#T_ee5c2_row13_col1, #T_ee5c2_row29_col3 {
  background-color: #91b5d6;
  color: #000000;
}
#T_ee5c2_row13_col22, #T_ee5c2_row15_col23, #T_ee5c2_row16_col2 {
  background-color: #acc0dd;
  color: #000000;
}
#T_ee5c2_row14_col0 {
  background-color: #f0eaf4;
  color: #000000;
}
#T_ee5c2_row14_col1, #T_ee5c2_row29_col1 {
  background-color: #f7f0f7;
  color: #000000;
}
#T_ee5c2_row14_col4 {
  background-color: #034d79;
  color: #f1f1f1;
}
#T_ee5c2_row14_col6, #T_ee5c2_row14_col14, #T_ee5c2_row24_col11, #T_ee5c2_row27_col11, #T_ee5c2_row28_col8, #T_ee5c2_row28_col14 {
  background-color: #04639b;
  color: #f1f1f1;
}
#T_ee5c2_row14_col11, #T_ee5c2_row19_col3, #T_ee5c2_row24_col21 {
  background-color: #04629a;
  color: #f1f1f1;
}
#T_ee5c2_row14_col13, #T_ee5c2_row15_col19, #T_ee5c2_row26_col0 {
  background-color: #034973;
  color: #f1f1f1;
}
#T_ee5c2_row14_col18, #T_ee5c2_row15_col5, #T_ee5c2_row16_col21 {
  background-color: #045788;
  color: #f1f1f1;
}
#T_ee5c2_row14_col22 {
  background-color: #97b7d7;
  color: #000000;
}
#T_ee5c2_row15_col0, #T_ee5c2_row22_col0 {
  background-color: #eee8f3;
  color: #000000;
}
#T_ee5c2_row15_col1, #T_ee5c2_row22_col1 {
  background-color: #f2ecf5;
  color: #000000;
}
#T_ee5c2_row15_col2, #T_ee5c2_row29_col0 {
  background-color: #f5eef6;
  color: #000000;
}
#T_ee5c2_row15_col7, #T_ee5c2_row15_col8, #T_ee5c2_row24_col9 {
  background-color: #023d60;
  color: #f1f1f1;
}
#T_ee5c2_row15_col9 {
  background-color: #023f64;
  color: #f1f1f1;
}
#T_ee5c2_row15_col17 {
  background-color: #03466e;
  color: #f1f1f1;
}
#T_ee5c2_row15_col21, #T_ee5c2_row24_col4 {
  background-color: #023b5d;
  color: #f1f1f1;
}
#T_ee5c2_row16_col1 {
  background-color: #c2cbe2;
  color: #000000;
}
#T_ee5c2_row16_col3, #T_ee5c2_row19_col7 {
  background-color: #73a9cf;
  color: #f1f1f1;
}
#T_ee5c2_row16_col5, #T_ee5c2_row16_col14, #T_ee5c2_row17_col13, #T_ee5c2_row17_col21, #T_ee5c2_row19_col14, #T_ee5c2_row27_col10 {
  background-color: #056aa6;
  color: #f1f1f1;
}
#T_ee5c2_row16_col12, #T_ee5c2_row16_col20, #T_ee5c2_row21_col13 {
  background-color: #056caa;
  color: #f1f1f1;
}
#T_ee5c2_row18_col0 {
  background-color: #b8c6e0;
  color: #000000;
}
#T_ee5c2_row18_col1 {
  background-color: #adc1dd;
  color: #000000;
}
#T_ee5c2_row19_col5, #T_ee5c2_row19_col8 {
  background-color: #6da6cd;
  color: #f1f1f1;
}
#T_ee5c2_row19_col6, #T_ee5c2_row23_col22 {
  background-color: #7dacd1;
  color: #f1f1f1;
}
#T_ee5c2_row20_col23, #T_ee5c2_row23_col0 {
  background-color: #b9c6e0;
  color: #000000;
}
#T_ee5c2_row21_col0, #T_ee5c2_row29_col23 {
  background-color: #dedcec;
  color: #000000;
}
#T_ee5c2_row21_col1 {
  background-color: #efe9f3;
  color: #000000;
}
#T_ee5c2_row21_col23 {
  background-color: #e8e4f0;
  color: #000000;
}
#T_ee5c2_row22_col3 {
  background-color: #89b1d4;
  color: #000000;
}
#T_ee5c2_row22_col5 {
  background-color: #034369;
  color: #f1f1f1;
}
#T_ee5c2_row23_col10, #T_ee5c2_row24_col22 {
  background-color: #197db7;
  color: #f1f1f1;
}
#T_ee5c2_row23_col23 {
  background-color: #ede7f2;
  color: #000000;
}
#T_ee5c2_row24_col2 {
  background-color: #faf3f9;
  color: #000000;
}
#T_ee5c2_row24_col3 {
  background-color: #84b0d3;
  color: #f1f1f1;
}
#T_ee5c2_row25_col7 {
  background-color: #5c9fc9;
  color: #f1f1f1;
}
#T_ee5c2_row26_col6, #T_ee5c2_row26_col20 {
  background-color: #81aed2;
  color: #f1f1f1;
}
#T_ee5c2_row26_col11, #T_ee5c2_row26_col13 {
  background-color: #045585;
  color: #f1f1f1;
}
#T_ee5c2_row27_col22 {
  background-color: #cdd0e5;
  color: #000000;
}
#T_ee5c2_row28_col0 {
  background-color: #e6e2ef;
  color: #000000;
}
#T_ee5c2_row28_col1 {
  background-color: #dddbec;
  color: #000000;
}
#T_ee5c2_row28_col5 {
  background-color: #034c78;
  color: #f1f1f1;
}
#T_ee5c2_row28_col23 {
  background-color: #f1ebf5;
  color: #000000;
}
#T_ee5c2_row29_col2 {
  background-color: #f6eff7;
  color: #000000;
}
#T_ee5c2_row29_col4 {
  background-color: #045382;
  color: #f1f1f1;
}
#T_ee5c2_row29_col6 {
  background-color: #023a5b;
  color: #f1f1f1;
}
#T_ee5c2_row29_col20 {
  background-color: #03517e;
  color: #f1f1f1;
}
</style>
<table id="T_ee5c2">
  <thead>
    <tr>
      <th class="index_name level0" >hour</th>
      <th id="T_ee5c2_level0_col0" class="col_heading level0 col0" >0</th>
      <th id="T_ee5c2_level0_col1" class="col_heading level0 col1" >1</th>
      <th id="T_ee5c2_level0_col2" class="col_heading level0 col2" >2</th>
      <th id="T_ee5c2_level0_col3" class="col_heading level0 col3" >3</th>
      <th id="T_ee5c2_level0_col4" class="col_heading level0 col4" >4</th>
      <th id="T_ee5c2_level0_col5" class="col_heading level0 col5" >5</th>
      <th id="T_ee5c2_level0_col6" class="col_heading level0 col6" >6</th>
      <th id="T_ee5c2_level0_col7" class="col_heading level0 col7" >7</th>
      <th id="T_ee5c2_level0_col8" class="col_heading level0 col8" >8</th>
      <th id="T_ee5c2_level0_col9" class="col_heading level0 col9" >9</th>
      <th id="T_ee5c2_level0_col10" class="col_heading level0 col10" >10</th>
      <th id="T_ee5c2_level0_col11" class="col_heading level0 col11" >11</th>
      <th id="T_ee5c2_level0_col12" class="col_heading level0 col12" >12</th>
      <th id="T_ee5c2_level0_col13" class="col_heading level0 col13" >13</th>
      <th id="T_ee5c2_level0_col14" class="col_heading level0 col14" >14</th>
      <th id="T_ee5c2_level0_col15" class="col_heading level0 col15" >15</th>
      <th id="T_ee5c2_level0_col16" class="col_heading level0 col16" >16</th>
      <th id="T_ee5c2_level0_col17" class="col_heading level0 col17" >17</th>
      <th id="T_ee5c2_level0_col18" class="col_heading level0 col18" >18</th>
      <th id="T_ee5c2_level0_col19" class="col_heading level0 col19" >19</th>
      <th id="T_ee5c2_level0_col20" class="col_heading level0 col20" >20</th>
      <th id="T_ee5c2_level0_col21" class="col_heading level0 col21" >21</th>
      <th id="T_ee5c2_level0_col22" class="col_heading level0 col22" >22</th>
      <th id="T_ee5c2_level0_col23" class="col_heading level0 col23" >23</th>
    </tr>
    <tr>
      <th class="index_name level0" >day</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
      <th class="blank col4" >&nbsp;</th>
      <th class="blank col5" >&nbsp;</th>
      <th class="blank col6" >&nbsp;</th>
      <th class="blank col7" >&nbsp;</th>
      <th class="blank col8" >&nbsp;</th>
      <th class="blank col9" >&nbsp;</th>
      <th class="blank col10" >&nbsp;</th>
      <th class="blank col11" >&nbsp;</th>
      <th class="blank col12" >&nbsp;</th>
      <th class="blank col13" >&nbsp;</th>
      <th class="blank col14" >&nbsp;</th>
      <th class="blank col15" >&nbsp;</th>
      <th class="blank col16" >&nbsp;</th>
      <th class="blank col17" >&nbsp;</th>
      <th class="blank col18" >&nbsp;</th>
      <th class="blank col19" >&nbsp;</th>
      <th class="blank col20" >&nbsp;</th>
      <th class="blank col21" >&nbsp;</th>
      <th class="blank col22" >&nbsp;</th>
      <th class="blank col23" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_ee5c2_level0_row0" class="row_heading level0 row0" >1</th>
      <td id="T_ee5c2_row0_col0" class="data row0 col0" >3178</td>
      <td id="T_ee5c2_row0_col1" class="data row0 col1" >1944</td>
      <td id="T_ee5c2_row0_col2" class="data row0 col2" >1256</td>
      <td id="T_ee5c2_row0_col3" class="data row0 col3" >1308</td>
      <td id="T_ee5c2_row0_col4" class="data row0 col4" >1429</td>
      <td id="T_ee5c2_row0_col5" class="data row0 col5" >2126</td>
      <td id="T_ee5c2_row0_col6" class="data row0 col6" >3664</td>
      <td id="T_ee5c2_row0_col7" class="data row0 col7" >5380</td>
      <td id="T_ee5c2_row0_col8" class="data row0 col8" >5292</td>
      <td id="T_ee5c2_row0_col9" class="data row0 col9" >4617</td>
      <td id="T_ee5c2_row0_col10" class="data row0 col10" >4607</td>
      <td id="T_ee5c2_row0_col11" class="data row0 col11" >4729</td>
      <td id="T_ee5c2_row0_col12" class="data row0 col12" >4930</td>
      <td id="T_ee5c2_row0_col13" class="data row0 col13" >5794</td>
      <td id="T_ee5c2_row0_col14" class="data row0 col14" >6933</td>
      <td id="T_ee5c2_row0_col15" class="data row0 col15" >7910</td>
      <td id="T_ee5c2_row0_col16" class="data row0 col16" >8633</td>
      <td id="T_ee5c2_row0_col17" class="data row0 col17" >9511</td>
      <td id="T_ee5c2_row0_col18" class="data row0 col18" >8604</td>
      <td id="T_ee5c2_row0_col19" class="data row0 col19" >8001</td>
      <td id="T_ee5c2_row0_col20" class="data row0 col20" >7315</td>
      <td id="T_ee5c2_row0_col21" class="data row0 col21" >7803</td>
      <td id="T_ee5c2_row0_col22" class="data row0 col22" >6268</td>
      <td id="T_ee5c2_row0_col23" class="data row0 col23" >4050</td>
    </tr>
    <tr>
      <th id="T_ee5c2_level0_row1" class="row_heading level0 row1" >2</th>
      <td id="T_ee5c2_row1_col0" class="data row1 col0" >2435</td>
      <td id="T_ee5c2_row1_col1" class="data row1 col1" >1569</td>
      <td id="T_ee5c2_row1_col2" class="data row1 col2" >1087</td>
      <td id="T_ee5c2_row1_col3" class="data row1 col3" >1414</td>
      <td id="T_ee5c2_row1_col4" class="data row1 col4" >1876</td>
      <td id="T_ee5c2_row1_col5" class="data row1 col5" >2812</td>
      <td id="T_ee5c2_row1_col6" class="data row1 col6" >4920</td>
      <td id="T_ee5c2_row1_col7" class="data row1 col7" >6544</td>
      <td id="T_ee5c2_row1_col8" class="data row1 col8" >6310</td>
      <td id="T_ee5c2_row1_col9" class="data row1 col9" >4712</td>
      <td id="T_ee5c2_row1_col10" class="data row1 col10" >4797</td>
      <td id="T_ee5c2_row1_col11" class="data row1 col11" >4975</td>
      <td id="T_ee5c2_row1_col12" class="data row1 col12" >5188</td>
      <td id="T_ee5c2_row1_col13" class="data row1 col13" >5695</td>
      <td id="T_ee5c2_row1_col14" class="data row1 col14" >6904</td>
      <td id="T_ee5c2_row1_col15" class="data row1 col15" >8449</td>
      <td id="T_ee5c2_row1_col16" class="data row1 col16" >10109</td>
      <td id="T_ee5c2_row1_col17" class="data row1 col17" >11100</td>
      <td id="T_ee5c2_row1_col18" class="data row1 col18" >11123</td>
      <td id="T_ee5c2_row1_col19" class="data row1 col19" >9474</td>
      <td id="T_ee5c2_row1_col20" class="data row1 col20" >8759</td>
      <td id="T_ee5c2_row1_col21" class="data row1 col21" >8357</td>
      <td id="T_ee5c2_row1_col22" class="data row1 col22" >6998</td>
      <td id="T_ee5c2_row1_col23" class="data row1 col23" >5160</td>
    </tr>
    <tr>
      <th id="T_ee5c2_level0_row2" class="row_heading level0 row2" >3</th>
      <td id="T_ee5c2_row2_col0" class="data row2 col0" >3354</td>
      <td id="T_ee5c2_row2_col1" class="data row2 col1" >2142</td>
      <td id="T_ee5c2_row2_col2" class="data row2 col2" >1407</td>
      <td id="T_ee5c2_row2_col3" class="data row2 col3" >1467</td>
      <td id="T_ee5c2_row2_col4" class="data row2 col4" >1550</td>
      <td id="T_ee5c2_row2_col5" class="data row2 col5" >2387</td>
      <td id="T_ee5c2_row2_col6" class="data row2 col6" >4241</td>
      <td id="T_ee5c2_row2_col7" class="data row2 col7" >5663</td>
      <td id="T_ee5c2_row2_col8" class="data row2 col8" >5386</td>
      <td id="T_ee5c2_row2_col9" class="data row2 col9" >4657</td>
      <td id="T_ee5c2_row2_col10" class="data row2 col10" >4788</td>
      <td id="T_ee5c2_row2_col11" class="data row2 col11" >5065</td>
      <td id="T_ee5c2_row2_col12" class="data row2 col12" >5384</td>
      <td id="T_ee5c2_row2_col13" class="data row2 col13" >6093</td>
      <td id="T_ee5c2_row2_col14" class="data row2 col14" >7226</td>
      <td id="T_ee5c2_row2_col15" class="data row2 col15" >8850</td>
      <td id="T_ee5c2_row2_col16" class="data row2 col16" >10314</td>
      <td id="T_ee5c2_row2_col17" class="data row2 col17" >10491</td>
      <td id="T_ee5c2_row2_col18" class="data row2 col18" >11239</td>
      <td id="T_ee5c2_row2_col19" class="data row2 col19" >9599</td>
      <td id="T_ee5c2_row2_col20" class="data row2 col20" >9026</td>
      <td id="T_ee5c2_row2_col21" class="data row2 col21" >8531</td>
      <td id="T_ee5c2_row2_col22" class="data row2 col22" >7142</td>
      <td id="T_ee5c2_row2_col23" class="data row2 col23" >4686</td>
    </tr>
    <tr>
      <th id="T_ee5c2_level0_row3" class="row_heading level0 row3" >4</th>
      <td id="T_ee5c2_row3_col0" class="data row3 col0" >2897</td>
      <td id="T_ee5c2_row3_col1" class="data row3 col1" >1688</td>
      <td id="T_ee5c2_row3_col2" class="data row3 col2" >1199</td>
      <td id="T_ee5c2_row3_col3" class="data row3 col3" >1424</td>
      <td id="T_ee5c2_row3_col4" class="data row3 col4" >1696</td>
      <td id="T_ee5c2_row3_col5" class="data row3 col5" >2581</td>
      <td id="T_ee5c2_row3_col6" class="data row3 col6" >4592</td>
      <td id="T_ee5c2_row3_col7" class="data row3 col7" >6029</td>
      <td id="T_ee5c2_row3_col8" class="data row3 col8" >5704</td>
      <td id="T_ee5c2_row3_col9" class="data row3 col9" >4744</td>
      <td id="T_ee5c2_row3_col10" class="data row3 col10" >4743</td>
      <td id="T_ee5c2_row3_col11" class="data row3 col11" >4975</td>
      <td id="T_ee5c2_row3_col12" class="data row3 col12" >5193</td>
      <td id="T_ee5c2_row3_col13" class="data row3 col13" >6175</td>
      <td id="T_ee5c2_row3_col14" class="data row3 col14" >7158</td>
      <td id="T_ee5c2_row3_col15" class="data row3 col15" >8515</td>
      <td id="T_ee5c2_row3_col16" class="data row3 col16" >9492</td>
      <td id="T_ee5c2_row3_col17" class="data row3 col17" >10357</td>
      <td id="T_ee5c2_row3_col18" class="data row3 col18" >10259</td>
      <td id="T_ee5c2_row3_col19" class="data row3 col19" >9097</td>
      <td id="T_ee5c2_row3_col20" class="data row3 col20" >8358</td>
      <td id="T_ee5c2_row3_col21" class="data row3 col21" >8649</td>
      <td id="T_ee5c2_row3_col22" class="data row3 col22" >7706</td>
      <td id="T_ee5c2_row3_col23" class="data row3 col23" >5130</td>
    </tr>
    <tr>
      <th id="T_ee5c2_level0_row4" class="row_heading level0 row4" >5</th>
      <td id="T_ee5c2_row4_col0" class="data row4 col0" >2733</td>
      <td id="T_ee5c2_row4_col1" class="data row4 col1" >1541</td>
      <td id="T_ee5c2_row4_col2" class="data row4 col2" >1030</td>
      <td id="T_ee5c2_row4_col3" class="data row4 col3" >1253</td>
      <td id="T_ee5c2_row4_col4" class="data row4 col4" >1617</td>
      <td id="T_ee5c2_row4_col5" class="data row4 col5" >2900</td>
      <td id="T_ee5c2_row4_col6" class="data row4 col6" >4814</td>
      <td id="T_ee5c2_row4_col7" class="data row4 col7" >6261</td>
      <td id="T_ee5c2_row4_col8" class="data row4 col8" >6469</td>
      <td id="T_ee5c2_row4_col9" class="data row4 col9" >5530</td>
      <td id="T_ee5c2_row4_col10" class="data row4 col10" >5141</td>
      <td id="T_ee5c2_row4_col11" class="data row4 col11" >5011</td>
      <td id="T_ee5c2_row4_col12" class="data row4 col12" >5047</td>
      <td id="T_ee5c2_row4_col13" class="data row4 col13" >5690</td>
      <td id="T_ee5c2_row4_col14" class="data row4 col14" >6955</td>
      <td id="T_ee5c2_row4_col15" class="data row4 col15" >8312</td>
      <td id="T_ee5c2_row4_col16" class="data row4 col16" >9609</td>
      <td id="T_ee5c2_row4_col17" class="data row4 col17" >10699</td>
      <td id="T_ee5c2_row4_col18" class="data row4 col18" >10170</td>
      <td id="T_ee5c2_row4_col19" class="data row4 col19" >9430</td>
      <td id="T_ee5c2_row4_col20" class="data row4 col20" >9354</td>
      <td id="T_ee5c2_row4_col21" class="data row4 col21" >9610</td>
      <td id="T_ee5c2_row4_col22" class="data row4 col22" >8853</td>
      <td id="T_ee5c2_row4_col23" class="data row4 col23" >6518</td>
    </tr>
    <tr>
      <th id="T_ee5c2_level0_row5" class="row_heading level0 row5" >6</th>
      <td id="T_ee5c2_row5_col0" class="data row5 col0" >4537</td>
      <td id="T_ee5c2_row5_col1" class="data row5 col1" >2864</td>
      <td id="T_ee5c2_row5_col2" class="data row5 col2" >1864</td>
      <td id="T_ee5c2_row5_col3" class="data row5 col3" >1555</td>
      <td id="T_ee5c2_row5_col4" class="data row5 col4" >1551</td>
      <td id="T_ee5c2_row5_col5" class="data row5 col5" >2162</td>
      <td id="T_ee5c2_row5_col6" class="data row5 col6" >3642</td>
      <td id="T_ee5c2_row5_col7" class="data row5 col7" >4766</td>
      <td id="T_ee5c2_row5_col8" class="data row5 col8" >4942</td>
      <td id="T_ee5c2_row5_col9" class="data row5 col9" >4401</td>
      <td id="T_ee5c2_row5_col10" class="data row5 col10" >4801</td>
      <td id="T_ee5c2_row5_col11" class="data row5 col11" >5174</td>
      <td id="T_ee5c2_row5_col12" class="data row5 col12" >5426</td>
      <td id="T_ee5c2_row5_col13" class="data row5 col13" >6258</td>
      <td id="T_ee5c2_row5_col14" class="data row5 col14" >7235</td>
      <td id="T_ee5c2_row5_col15" class="data row5 col15" >8612</td>
      <td id="T_ee5c2_row5_col16" class="data row5 col16" >9444</td>
      <td id="T_ee5c2_row5_col17" class="data row5 col17" >9929</td>
      <td id="T_ee5c2_row5_col18" class="data row5 col18" >9263</td>
      <td id="T_ee5c2_row5_col19" class="data row5 col19" >8405</td>
      <td id="T_ee5c2_row5_col20" class="data row5 col20" >8117</td>
      <td id="T_ee5c2_row5_col21" class="data row5 col21" >8567</td>
      <td id="T_ee5c2_row5_col22" class="data row5 col22" >7852</td>
      <td id="T_ee5c2_row5_col23" class="data row5 col23" >5946</td>
    </tr>
    <tr>
      <th id="T_ee5c2_level0_row6" class="row_heading level0 row6" >7</th>
      <td id="T_ee5c2_row6_col0" class="data row6 col0" >3645</td>
      <td id="T_ee5c2_row6_col1" class="data row6 col1" >2296</td>
      <td id="T_ee5c2_row6_col2" class="data row6 col2" >1507</td>
      <td id="T_ee5c2_row6_col3" class="data row6 col3" >1597</td>
      <td id="T_ee5c2_row6_col4" class="data row6 col4" >1763</td>
      <td id="T_ee5c2_row6_col5" class="data row6 col5" >2422</td>
      <td id="T_ee5c2_row6_col6" class="data row6 col6" >4102</td>
      <td id="T_ee5c2_row6_col7" class="data row6 col7" >5575</td>
      <td id="T_ee5c2_row6_col8" class="data row6 col8" >5376</td>
      <td id="T_ee5c2_row6_col9" class="data row6 col9" >4639</td>
      <td id="T_ee5c2_row6_col10" class="data row6 col10" >4905</td>
      <td id="T_ee5c2_row6_col11" class="data row6 col11" >5166</td>
      <td id="T_ee5c2_row6_col12" class="data row6 col12" >5364</td>
      <td id="T_ee5c2_row6_col13" class="data row6 col13" >6214</td>
      <td id="T_ee5c2_row6_col14" class="data row6 col14" >7276</td>
      <td id="T_ee5c2_row6_col15" class="data row6 col15" >8474</td>
      <td id="T_ee5c2_row6_col16" class="data row6 col16" >10393</td>
      <td id="T_ee5c2_row6_col17" class="data row6 col17" >11013</td>
      <td id="T_ee5c2_row6_col18" class="data row6 col18" >10573</td>
      <td id="T_ee5c2_row6_col19" class="data row6 col19" >9472</td>
      <td id="T_ee5c2_row6_col20" class="data row6 col20" >8691</td>
      <td id="T_ee5c2_row6_col21" class="data row6 col21" >8525</td>
      <td id="T_ee5c2_row6_col22" class="data row6 col22" >7194</td>
      <td id="T_ee5c2_row6_col23" class="data row6 col23" >4801</td>
    </tr>
    <tr>
      <th id="T_ee5c2_level0_row7" class="row_heading level0 row7" >8</th>
      <td id="T_ee5c2_row7_col0" class="data row7 col0" >2830</td>
      <td id="T_ee5c2_row7_col1" class="data row7 col1" >1646</td>
      <td id="T_ee5c2_row7_col2" class="data row7 col2" >1123</td>
      <td id="T_ee5c2_row7_col3" class="data row7 col3" >1483</td>
      <td id="T_ee5c2_row7_col4" class="data row7 col4" >1889</td>
      <td id="T_ee5c2_row7_col5" class="data row7 col5" >3224</td>
      <td id="T_ee5c2_row7_col6" class="data row7 col6" >5431</td>
      <td id="T_ee5c2_row7_col7" class="data row7 col7" >7361</td>
      <td id="T_ee5c2_row7_col8" class="data row7 col8" >7357</td>
      <td id="T_ee5c2_row7_col9" class="data row7 col9" >5703</td>
      <td id="T_ee5c2_row7_col10" class="data row7 col10" >5288</td>
      <td id="T_ee5c2_row7_col11" class="data row7 col11" >5350</td>
      <td id="T_ee5c2_row7_col12" class="data row7 col12" >5483</td>
      <td id="T_ee5c2_row7_col13" class="data row7 col13" >6318</td>
      <td id="T_ee5c2_row7_col14" class="data row7 col14" >7240</td>
      <td id="T_ee5c2_row7_col15" class="data row7 col15" >8775</td>
      <td id="T_ee5c2_row7_col16" class="data row7 col16" >9851</td>
      <td id="T_ee5c2_row7_col17" class="data row7 col17" >10673</td>
      <td id="T_ee5c2_row7_col18" class="data row7 col18" >9687</td>
      <td id="T_ee5c2_row7_col19" class="data row7 col19" >8796</td>
      <td id="T_ee5c2_row7_col20" class="data row7 col20" >8604</td>
      <td id="T_ee5c2_row7_col21" class="data row7 col21" >8367</td>
      <td id="T_ee5c2_row7_col22" class="data row7 col22" >6795</td>
      <td id="T_ee5c2_row7_col23" class="data row7 col23" >4256</td>
    </tr>
    <tr>
      <th id="T_ee5c2_level0_row8" class="row_heading level0 row8" >9</th>
      <td id="T_ee5c2_row8_col0" class="data row8 col0" >2657</td>
      <td id="T_ee5c2_row8_col1" class="data row8 col1" >1724</td>
      <td id="T_ee5c2_row8_col2" class="data row8 col2" >1222</td>
      <td id="T_ee5c2_row8_col3" class="data row8 col3" >1480</td>
      <td id="T_ee5c2_row8_col4" class="data row8 col4" >1871</td>
      <td id="T_ee5c2_row8_col5" class="data row8 col5" >3168</td>
      <td id="T_ee5c2_row8_col6" class="data row8 col6" >5802</td>
      <td id="T_ee5c2_row8_col7" class="data row8 col7" >7592</td>
      <td id="T_ee5c2_row8_col8" class="data row8 col8" >7519</td>
      <td id="T_ee5c2_row8_col9" class="data row8 col9" >5895</td>
      <td id="T_ee5c2_row8_col10" class="data row8 col10" >5406</td>
      <td id="T_ee5c2_row8_col11" class="data row8 col11" >5443</td>
      <td id="T_ee5c2_row8_col12" class="data row8 col12" >5496</td>
      <td id="T_ee5c2_row8_col13" class="data row8 col13" >6419</td>
      <td id="T_ee5c2_row8_col14" class="data row8 col14" >7877</td>
      <td id="T_ee5c2_row8_col15" class="data row8 col15" >9220</td>
      <td id="T_ee5c2_row8_col16" class="data row8 col16" >10270</td>
      <td id="T_ee5c2_row8_col17" class="data row8 col17" >11910</td>
      <td id="T_ee5c2_row8_col18" class="data row8 col18" >11449</td>
      <td id="T_ee5c2_row8_col19" class="data row8 col19" >9804</td>
      <td id="T_ee5c2_row8_col20" class="data row8 col20" >8909</td>
      <td id="T_ee5c2_row8_col21" class="data row8 col21" >8665</td>
      <td id="T_ee5c2_row8_col22" class="data row8 col22" >7499</td>
      <td id="T_ee5c2_row8_col23" class="data row8 col23" >5203</td>
    </tr>
    <tr>
      <th id="T_ee5c2_level0_row9" class="row_heading level0 row9" >10</th>
      <td id="T_ee5c2_row9_col0" class="data row9 col0" >3296</td>
      <td id="T_ee5c2_row9_col1" class="data row9 col1" >2126</td>
      <td id="T_ee5c2_row9_col2" class="data row9 col2" >1464</td>
      <td id="T_ee5c2_row9_col3" class="data row9 col3" >1434</td>
      <td id="T_ee5c2_row9_col4" class="data row9 col4" >1591</td>
      <td id="T_ee5c2_row9_col5" class="data row9 col5" >2594</td>
      <td id="T_ee5c2_row9_col6" class="data row9 col6" >4664</td>
      <td id="T_ee5c2_row9_col7" class="data row9 col7" >6046</td>
      <td id="T_ee5c2_row9_col8" class="data row9 col8" >6158</td>
      <td id="T_ee5c2_row9_col9" class="data row9 col9" >5072</td>
      <td id="T_ee5c2_row9_col10" class="data row9 col10" >4976</td>
      <td id="T_ee5c2_row9_col11" class="data row9 col11" >5415</td>
      <td id="T_ee5c2_row9_col12" class="data row9 col12" >5506</td>
      <td id="T_ee5c2_row9_col13" class="data row9 col13" >6527</td>
      <td id="T_ee5c2_row9_col14" class="data row9 col14" >7612</td>
      <td id="T_ee5c2_row9_col15" class="data row9 col15" >9578</td>
      <td id="T_ee5c2_row9_col16" class="data row9 col16" >11045</td>
      <td id="T_ee5c2_row9_col17" class="data row9 col17" >11875</td>
      <td id="T_ee5c2_row9_col18" class="data row9 col18" >10934</td>
      <td id="T_ee5c2_row9_col19" class="data row9 col19" >9613</td>
      <td id="T_ee5c2_row9_col20" class="data row9 col20" >9687</td>
      <td id="T_ee5c2_row9_col21" class="data row9 col21" >9240</td>
      <td id="T_ee5c2_row9_col22" class="data row9 col22" >7766</td>
      <td id="T_ee5c2_row9_col23" class="data row9 col23" >5496</td>
    </tr>
    <tr>
      <th id="T_ee5c2_level0_row10" class="row_heading level0 row10" >11</th>
      <td id="T_ee5c2_row10_col0" class="data row10 col0" >3036</td>
      <td id="T_ee5c2_row10_col1" class="data row10 col1" >1665</td>
      <td id="T_ee5c2_row10_col2" class="data row10 col2" >1095</td>
      <td id="T_ee5c2_row10_col3" class="data row10 col3" >1424</td>
      <td id="T_ee5c2_row10_col4" class="data row10 col4" >1842</td>
      <td id="T_ee5c2_row10_col5" class="data row10 col5" >2520</td>
      <td id="T_ee5c2_row10_col6" class="data row10 col6" >4954</td>
      <td id="T_ee5c2_row10_col7" class="data row10 col7" >6876</td>
      <td id="T_ee5c2_row10_col8" class="data row10 col8" >6871</td>
      <td id="T_ee5c2_row10_col9" class="data row10 col9" >5396</td>
      <td id="T_ee5c2_row10_col10" class="data row10 col10" >5215</td>
      <td id="T_ee5c2_row10_col11" class="data row10 col11" >5423</td>
      <td id="T_ee5c2_row10_col12" class="data row10 col12" >5513</td>
      <td id="T_ee5c2_row10_col13" class="data row10 col13" >6486</td>
      <td id="T_ee5c2_row10_col14" class="data row10 col14" >7503</td>
      <td id="T_ee5c2_row10_col15" class="data row10 col15" >8920</td>
      <td id="T_ee5c2_row10_col16" class="data row10 col16" >10125</td>
      <td id="T_ee5c2_row10_col17" class="data row10 col17" >10898</td>
      <td id="T_ee5c2_row10_col18" class="data row10 col18" >10361</td>
      <td id="T_ee5c2_row10_col19" class="data row10 col19" >9327</td>
      <td id="T_ee5c2_row10_col20" class="data row10 col20" >8824</td>
      <td id="T_ee5c2_row10_col21" class="data row10 col21" >8730</td>
      <td id="T_ee5c2_row10_col22" class="data row10 col22" >7771</td>
      <td id="T_ee5c2_row10_col23" class="data row10 col23" >5360</td>
    </tr>
    <tr>
      <th id="T_ee5c2_level0_row11" class="row_heading level0 row11" >12</th>
      <td id="T_ee5c2_row11_col0" class="data row11 col0" >3227</td>
      <td id="T_ee5c2_row11_col1" class="data row11 col1" >2147</td>
      <td id="T_ee5c2_row11_col2" class="data row11 col2" >1393</td>
      <td id="T_ee5c2_row11_col3" class="data row11 col3" >1362</td>
      <td id="T_ee5c2_row11_col4" class="data row11 col4" >1757</td>
      <td id="T_ee5c2_row11_col5" class="data row11 col5" >2710</td>
      <td id="T_ee5c2_row11_col6" class="data row11 col6" >4576</td>
      <td id="T_ee5c2_row11_col7" class="data row11 col7" >6250</td>
      <td id="T_ee5c2_row11_col8" class="data row11 col8" >6231</td>
      <td id="T_ee5c2_row11_col9" class="data row11 col9" >5177</td>
      <td id="T_ee5c2_row11_col10" class="data row11 col10" >5157</td>
      <td id="T_ee5c2_row11_col11" class="data row11 col11" >5319</td>
      <td id="T_ee5c2_row11_col12" class="data row11 col12" >5570</td>
      <td id="T_ee5c2_row11_col13" class="data row11 col13" >6448</td>
      <td id="T_ee5c2_row11_col14" class="data row11 col14" >7743</td>
      <td id="T_ee5c2_row11_col15" class="data row11 col15" >9390</td>
      <td id="T_ee5c2_row11_col16" class="data row11 col16" >10734</td>
      <td id="T_ee5c2_row11_col17" class="data row11 col17" >11713</td>
      <td id="T_ee5c2_row11_col18" class="data row11 col18" >12216</td>
      <td id="T_ee5c2_row11_col19" class="data row11 col19" >10393</td>
      <td id="T_ee5c2_row11_col20" class="data row11 col20" >9965</td>
      <td id="T_ee5c2_row11_col21" class="data row11 col21" >10310</td>
      <td id="T_ee5c2_row11_col22" class="data row11 col22" >9992</td>
      <td id="T_ee5c2_row11_col23" class="data row11 col23" >7945</td>
    </tr>
    <tr>
      <th id="T_ee5c2_level0_row12" class="row_heading level0 row12" >13</th>
      <td id="T_ee5c2_row12_col0" class="data row12 col0" >5408</td>
      <td id="T_ee5c2_row12_col1" class="data row12 col1" >3509</td>
      <td id="T_ee5c2_row12_col2" class="data row12 col2" >2262</td>
      <td id="T_ee5c2_row12_col3" class="data row12 col3" >1832</td>
      <td id="T_ee5c2_row12_col4" class="data row12 col4" >1705</td>
      <td id="T_ee5c2_row12_col5" class="data row12 col5" >2327</td>
      <td id="T_ee5c2_row12_col6" class="data row12 col6" >4196</td>
      <td id="T_ee5c2_row12_col7" class="data row12 col7" >5685</td>
      <td id="T_ee5c2_row12_col8" class="data row12 col8" >6060</td>
      <td id="T_ee5c2_row12_col9" class="data row12 col9" >5631</td>
      <td id="T_ee5c2_row12_col10" class="data row12 col10" >5442</td>
      <td id="T_ee5c2_row12_col11" class="data row12 col11" >5720</td>
      <td id="T_ee5c2_row12_col12" class="data row12 col12" >5914</td>
      <td id="T_ee5c2_row12_col13" class="data row12 col13" >6678</td>
      <td id="T_ee5c2_row12_col14" class="data row12 col14" >8200</td>
      <td id="T_ee5c2_row12_col15" class="data row12 col15" >9264</td>
      <td id="T_ee5c2_row12_col16" class="data row12 col16" >10534</td>
      <td id="T_ee5c2_row12_col17" class="data row12 col17" >11826</td>
      <td id="T_ee5c2_row12_col18" class="data row12 col18" >11450</td>
      <td id="T_ee5c2_row12_col19" class="data row12 col19" >9921</td>
      <td id="T_ee5c2_row12_col20" class="data row12 col20" >8705</td>
      <td id="T_ee5c2_row12_col21" class="data row12 col21" >8423</td>
      <td id="T_ee5c2_row12_col22" class="data row12 col22" >7363</td>
      <td id="T_ee5c2_row12_col23" class="data row12 col23" >5936</td>
    </tr>
    <tr>
      <th id="T_ee5c2_level0_row13" class="row_heading level0 row13" >14</th>
      <td id="T_ee5c2_row13_col0" class="data row13 col0" >3748</td>
      <td id="T_ee5c2_row13_col1" class="data row13 col1" >2349</td>
      <td id="T_ee5c2_row13_col2" class="data row13 col2" >1605</td>
      <td id="T_ee5c2_row13_col3" class="data row13 col3" >1656</td>
      <td id="T_ee5c2_row13_col4" class="data row13 col4" >1756</td>
      <td id="T_ee5c2_row13_col5" class="data row13 col5" >2629</td>
      <td id="T_ee5c2_row13_col6" class="data row13 col6" >4257</td>
      <td id="T_ee5c2_row13_col7" class="data row13 col7" >5781</td>
      <td id="T_ee5c2_row13_col8" class="data row13 col8" >5520</td>
      <td id="T_ee5c2_row13_col9" class="data row13 col9" >4824</td>
      <td id="T_ee5c2_row13_col10" class="data row13 col10" >4911</td>
      <td id="T_ee5c2_row13_col11" class="data row13 col11" >5118</td>
      <td id="T_ee5c2_row13_col12" class="data row13 col12" >5153</td>
      <td id="T_ee5c2_row13_col13" class="data row13 col13" >5747</td>
      <td id="T_ee5c2_row13_col14" class="data row13 col14" >6963</td>
      <td id="T_ee5c2_row13_col15" class="data row13 col15" >8192</td>
      <td id="T_ee5c2_row13_col16" class="data row13 col16" >9511</td>
      <td id="T_ee5c2_row13_col17" class="data row13 col17" >10115</td>
      <td id="T_ee5c2_row13_col18" class="data row13 col18" >9553</td>
      <td id="T_ee5c2_row13_col19" class="data row13 col19" >9146</td>
      <td id="T_ee5c2_row13_col20" class="data row13 col20" >9182</td>
      <td id="T_ee5c2_row13_col21" class="data row13 col21" >8589</td>
      <td id="T_ee5c2_row13_col22" class="data row13 col22" >6891</td>
      <td id="T_ee5c2_row13_col23" class="data row13 col23" >4460</td>
    </tr>
    <tr>
      <th id="T_ee5c2_level0_row14" class="row_heading level0 row14" >15</th>
      <td id="T_ee5c2_row14_col0" class="data row14 col0" >2497</td>
      <td id="T_ee5c2_row14_col1" class="data row14 col1" >1515</td>
      <td id="T_ee5c2_row14_col2" class="data row14 col2" >1087</td>
      <td id="T_ee5c2_row14_col3" class="data row14 col3" >1381</td>
      <td id="T_ee5c2_row14_col4" class="data row14 col4" >1862</td>
      <td id="T_ee5c2_row14_col5" class="data row14 col5" >2980</td>
      <td id="T_ee5c2_row14_col6" class="data row14 col6" >5050</td>
      <td id="T_ee5c2_row14_col7" class="data row14 col7" >6837</td>
      <td id="T_ee5c2_row14_col8" class="data row14 col8" >6729</td>
      <td id="T_ee5c2_row14_col9" class="data row14 col9" >5201</td>
      <td id="T_ee5c2_row14_col10" class="data row14 col10" >5347</td>
      <td id="T_ee5c2_row14_col11" class="data row14 col11" >5517</td>
      <td id="T_ee5c2_row14_col12" class="data row14 col12" >5503</td>
      <td id="T_ee5c2_row14_col13" class="data row14 col13" >6997</td>
      <td id="T_ee5c2_row14_col14" class="data row14 col14" >7633</td>
      <td id="T_ee5c2_row14_col15" class="data row14 col15" >8505</td>
      <td id="T_ee5c2_row14_col16" class="data row14 col16" >10285</td>
      <td id="T_ee5c2_row14_col17" class="data row14 col17" >11959</td>
      <td id="T_ee5c2_row14_col18" class="data row14 col18" >11728</td>
      <td id="T_ee5c2_row14_col19" class="data row14 col19" >11032</td>
      <td id="T_ee5c2_row14_col20" class="data row14 col20" >10509</td>
      <td id="T_ee5c2_row14_col21" class="data row14 col21" >9105</td>
      <td id="T_ee5c2_row14_col22" class="data row14 col22" >7153</td>
      <td id="T_ee5c2_row14_col23" class="data row14 col23" >4480</td>
    </tr>
    <tr>
      <th id="T_ee5c2_level0_row15" class="row_heading level0 row15" >16</th>
      <td id="T_ee5c2_row15_col0" class="data row15 col0" >2547</td>
      <td id="T_ee5c2_row15_col1" class="data row15 col1" >1585</td>
      <td id="T_ee5c2_row15_col2" class="data row15 col2" >1119</td>
      <td id="T_ee5c2_row15_col3" class="data row15 col3" >1395</td>
      <td id="T_ee5c2_row15_col4" class="data row15 col4" >1818</td>
      <td id="T_ee5c2_row15_col5" class="data row15 col5" >2966</td>
      <td id="T_ee5c2_row15_col6" class="data row15 col6" >5558</td>
      <td id="T_ee5c2_row15_col7" class="data row15 col7" >7517</td>
      <td id="T_ee5c2_row15_col8" class="data row15 col8" >7495</td>
      <td id="T_ee5c2_row15_col9" class="data row15 col9" >5958</td>
      <td id="T_ee5c2_row15_col10" class="data row15 col10" >5626</td>
      <td id="T_ee5c2_row15_col11" class="data row15 col11" >5480</td>
      <td id="T_ee5c2_row15_col12" class="data row15 col12" >5525</td>
      <td id="T_ee5c2_row15_col13" class="data row15 col13" >6198</td>
      <td id="T_ee5c2_row15_col14" class="data row15 col14" >7597</td>
      <td id="T_ee5c2_row15_col15" class="data row15 col15" >9290</td>
      <td id="T_ee5c2_row15_col16" class="data row15 col16" >10804</td>
      <td id="T_ee5c2_row15_col17" class="data row15 col17" >11773</td>
      <td id="T_ee5c2_row15_col18" class="data row15 col18" >10855</td>
      <td id="T_ee5c2_row15_col19" class="data row15 col19" >10924</td>
      <td id="T_ee5c2_row15_col20" class="data row15 col20" >10142</td>
      <td id="T_ee5c2_row15_col21" class="data row15 col21" >10374</td>
      <td id="T_ee5c2_row15_col22" class="data row15 col22" >8094</td>
      <td id="T_ee5c2_row15_col23" class="data row15 col23" >5380</td>
    </tr>
    <tr>
      <th id="T_ee5c2_level0_row16" class="row_heading level0 row16" >17</th>
      <td id="T_ee5c2_row16_col0" class="data row16 col0" >3155</td>
      <td id="T_ee5c2_row16_col1" class="data row16 col1" >2048</td>
      <td id="T_ee5c2_row16_col2" class="data row16 col2" >1500</td>
      <td id="T_ee5c2_row16_col3" class="data row16 col3" >1488</td>
      <td id="T_ee5c2_row16_col4" class="data row16 col4" >1897</td>
      <td id="T_ee5c2_row16_col5" class="data row16 col5" >2741</td>
      <td id="T_ee5c2_row16_col6" class="data row16 col6" >4562</td>
      <td id="T_ee5c2_row16_col7" class="data row16 col7" >6315</td>
      <td id="T_ee5c2_row16_col8" class="data row16 col8" >5882</td>
      <td id="T_ee5c2_row16_col9" class="data row16 col9" >4934</td>
      <td id="T_ee5c2_row16_col10" class="data row16 col10" >5004</td>
      <td id="T_ee5c2_row16_col11" class="data row16 col11" >5306</td>
      <td id="T_ee5c2_row16_col12" class="data row16 col12" >5634</td>
      <td id="T_ee5c2_row16_col13" class="data row16 col13" >6507</td>
      <td id="T_ee5c2_row16_col14" class="data row16 col14" >7472</td>
      <td id="T_ee5c2_row16_col15" class="data row16 col15" >8997</td>
      <td id="T_ee5c2_row16_col16" class="data row16 col16" >10323</td>
      <td id="T_ee5c2_row16_col17" class="data row16 col17" >11236</td>
      <td id="T_ee5c2_row16_col18" class="data row16 col18" >11089</td>
      <td id="T_ee5c2_row16_col19" class="data row16 col19" >9919</td>
      <td id="T_ee5c2_row16_col20" class="data row16 col20" >9935</td>
      <td id="T_ee5c2_row16_col21" class="data row16 col21" >9823</td>
      <td id="T_ee5c2_row16_col22" class="data row16 col22" >8362</td>
      <td id="T_ee5c2_row16_col23" class="data row16 col23" >5699</td>
    </tr>
    <tr>
      <th id="T_ee5c2_level0_row17" class="row_heading level0 row17" >18</th>
      <td id="T_ee5c2_row17_col0" class="data row17 col0" >3390</td>
      <td id="T_ee5c2_row17_col1" class="data row17 col1" >2135</td>
      <td id="T_ee5c2_row17_col2" class="data row17 col2" >1332</td>
      <td id="T_ee5c2_row17_col3" class="data row17 col3" >1626</td>
      <td id="T_ee5c2_row17_col4" class="data row17 col4" >1892</td>
      <td id="T_ee5c2_row17_col5" class="data row17 col5" >2959</td>
      <td id="T_ee5c2_row17_col6" class="data row17 col6" >4688</td>
      <td id="T_ee5c2_row17_col7" class="data row17 col7" >6618</td>
      <td id="T_ee5c2_row17_col8" class="data row17 col8" >6451</td>
      <td id="T_ee5c2_row17_col9" class="data row17 col9" >5377</td>
      <td id="T_ee5c2_row17_col10" class="data row17 col10" >5150</td>
      <td id="T_ee5c2_row17_col11" class="data row17 col11" >5487</td>
      <td id="T_ee5c2_row17_col12" class="data row17 col12" >5490</td>
      <td id="T_ee5c2_row17_col13" class="data row17 col13" >6383</td>
      <td id="T_ee5c2_row17_col14" class="data row17 col14" >7534</td>
      <td id="T_ee5c2_row17_col15" class="data row17 col15" >9040</td>
      <td id="T_ee5c2_row17_col16" class="data row17 col16" >10274</td>
      <td id="T_ee5c2_row17_col17" class="data row17 col17" >10692</td>
      <td id="T_ee5c2_row17_col18" class="data row17 col18" >10338</td>
      <td id="T_ee5c2_row17_col19" class="data row17 col19" >9551</td>
      <td id="T_ee5c2_row17_col20" class="data row17 col20" >9310</td>
      <td id="T_ee5c2_row17_col21" class="data row17 col21" >9285</td>
      <td id="T_ee5c2_row17_col22" class="data row17 col22" >8015</td>
      <td id="T_ee5c2_row17_col23" class="data row17 col23" >5492</td>
    </tr>
    <tr>
      <th id="T_ee5c2_level0_row18" class="row_heading level0 row18" >19</th>
      <td id="T_ee5c2_row18_col0" class="data row18 col0" >3217</td>
      <td id="T_ee5c2_row18_col1" class="data row18 col1" >2188</td>
      <td id="T_ee5c2_row18_col2" class="data row18 col2" >1604</td>
      <td id="T_ee5c2_row18_col3" class="data row18 col3" >1675</td>
      <td id="T_ee5c2_row18_col4" class="data row18 col4" >1810</td>
      <td id="T_ee5c2_row18_col5" class="data row18 col5" >2639</td>
      <td id="T_ee5c2_row18_col6" class="data row18 col6" >4733</td>
      <td id="T_ee5c2_row18_col7" class="data row18 col7" >6159</td>
      <td id="T_ee5c2_row18_col8" class="data row18 col8" >6014</td>
      <td id="T_ee5c2_row18_col9" class="data row18 col9" >5006</td>
      <td id="T_ee5c2_row18_col10" class="data row18 col10" >5092</td>
      <td id="T_ee5c2_row18_col11" class="data row18 col11" >5240</td>
      <td id="T_ee5c2_row18_col12" class="data row18 col12" >5590</td>
      <td id="T_ee5c2_row18_col13" class="data row18 col13" >6367</td>
      <td id="T_ee5c2_row18_col14" class="data row18 col14" >7374</td>
      <td id="T_ee5c2_row18_col15" class="data row18 col15" >8898</td>
      <td id="T_ee5c2_row18_col16" class="data row18 col16" >9893</td>
      <td id="T_ee5c2_row18_col17" class="data row18 col17" >10741</td>
      <td id="T_ee5c2_row18_col18" class="data row18 col18" >10429</td>
      <td id="T_ee5c2_row18_col19" class="data row18 col19" >9701</td>
      <td id="T_ee5c2_row18_col20" class="data row18 col20" >10051</td>
      <td id="T_ee5c2_row18_col21" class="data row18 col21" >10049</td>
      <td id="T_ee5c2_row18_col22" class="data row18 col22" >9090</td>
      <td id="T_ee5c2_row18_col23" class="data row18 col23" >6666</td>
    </tr>
    <tr>
      <th id="T_ee5c2_level0_row19" class="row_heading level0 row19" >20</th>
      <td id="T_ee5c2_row19_col0" class="data row19 col0" >4475</td>
      <td id="T_ee5c2_row19_col1" class="data row19 col1" >3190</td>
      <td id="T_ee5c2_row19_col2" class="data row19 col2" >2100</td>
      <td id="T_ee5c2_row19_col3" class="data row19 col3" >1858</td>
      <td id="T_ee5c2_row19_col4" class="data row19 col4" >1618</td>
      <td id="T_ee5c2_row19_col5" class="data row19 col5" >2143</td>
      <td id="T_ee5c2_row19_col6" class="data row19 col6" >3584</td>
      <td id="T_ee5c2_row19_col7" class="data row19 col7" >4900</td>
      <td id="T_ee5c2_row19_col8" class="data row19 col8" >5083</td>
      <td id="T_ee5c2_row19_col9" class="data row19 col9" >4765</td>
      <td id="T_ee5c2_row19_col10" class="data row19 col10" >5135</td>
      <td id="T_ee5c2_row19_col11" class="data row19 col11" >5650</td>
      <td id="T_ee5c2_row19_col12" class="data row19 col12" >5745</td>
      <td id="T_ee5c2_row19_col13" class="data row19 col13" >6656</td>
      <td id="T_ee5c2_row19_col14" class="data row19 col14" >7462</td>
      <td id="T_ee5c2_row19_col15" class="data row19 col15" >8630</td>
      <td id="T_ee5c2_row19_col16" class="data row19 col16" >9448</td>
      <td id="T_ee5c2_row19_col17" class="data row19 col17" >10046</td>
      <td id="T_ee5c2_row19_col18" class="data row19 col18" >9272</td>
      <td id="T_ee5c2_row19_col19" class="data row19 col19" >8592</td>
      <td id="T_ee5c2_row19_col20" class="data row19 col20" >8614</td>
      <td id="T_ee5c2_row19_col21" class="data row19 col21" >8703</td>
      <td id="T_ee5c2_row19_col22" class="data row19 col22" >7787</td>
      <td id="T_ee5c2_row19_col23" class="data row19 col23" >5907</td>
    </tr>
    <tr>
      <th id="T_ee5c2_level0_row20" class="row_heading level0 row20" >21</th>
      <td id="T_ee5c2_row20_col0" class="data row20 col0" >4294</td>
      <td id="T_ee5c2_row20_col1" class="data row20 col1" >3194</td>
      <td id="T_ee5c2_row20_col2" class="data row20 col2" >1972</td>
      <td id="T_ee5c2_row20_col3" class="data row20 col3" >1727</td>
      <td id="T_ee5c2_row20_col4" class="data row20 col4" >1926</td>
      <td id="T_ee5c2_row20_col5" class="data row20 col5" >2615</td>
      <td id="T_ee5c2_row20_col6" class="data row20 col6" >4185</td>
      <td id="T_ee5c2_row20_col7" class="data row20 col7" >5727</td>
      <td id="T_ee5c2_row20_col8" class="data row20 col8" >5529</td>
      <td id="T_ee5c2_row20_col9" class="data row20 col9" >4707</td>
      <td id="T_ee5c2_row20_col10" class="data row20 col10" >4911</td>
      <td id="T_ee5c2_row20_col11" class="data row20 col11" >5212</td>
      <td id="T_ee5c2_row20_col12" class="data row20 col12" >5465</td>
      <td id="T_ee5c2_row20_col13" class="data row20 col13" >6085</td>
      <td id="T_ee5c2_row20_col14" class="data row20 col14" >7064</td>
      <td id="T_ee5c2_row20_col15" class="data row20 col15" >8127</td>
      <td id="T_ee5c2_row20_col16" class="data row20 col16" >9483</td>
      <td id="T_ee5c2_row20_col17" class="data row20 col17" >9817</td>
      <td id="T_ee5c2_row20_col18" class="data row20 col18" >9291</td>
      <td id="T_ee5c2_row20_col19" class="data row20 col19" >8317</td>
      <td id="T_ee5c2_row20_col20" class="data row20 col20" >8107</td>
      <td id="T_ee5c2_row20_col21" class="data row20 col21" >8245</td>
      <td id="T_ee5c2_row20_col22" class="data row20 col22" >7362</td>
      <td id="T_ee5c2_row20_col23" class="data row20 col23" >5231</td>
    </tr>
    <tr>
      <th id="T_ee5c2_level0_row21" class="row_heading level0 row21" >22</th>
      <td id="T_ee5c2_row21_col0" class="data row21 col0" >2787</td>
      <td id="T_ee5c2_row21_col1" class="data row21 col1" >1637</td>
      <td id="T_ee5c2_row21_col2" class="data row21 col2" >1175</td>
      <td id="T_ee5c2_row21_col3" class="data row21 col3" >1468</td>
      <td id="T_ee5c2_row21_col4" class="data row21 col4" >1934</td>
      <td id="T_ee5c2_row21_col5" class="data row21 col5" >3151</td>
      <td id="T_ee5c2_row21_col6" class="data row21 col6" >5204</td>
      <td id="T_ee5c2_row21_col7" class="data row21 col7" >6872</td>
      <td id="T_ee5c2_row21_col8" class="data row21 col8" >6850</td>
      <td id="T_ee5c2_row21_col9" class="data row21 col9" >5198</td>
      <td id="T_ee5c2_row21_col10" class="data row21 col10" >5277</td>
      <td id="T_ee5c2_row21_col11" class="data row21 col11" >5352</td>
      <td id="T_ee5c2_row21_col12" class="data row21 col12" >5512</td>
      <td id="T_ee5c2_row21_col13" class="data row21 col13" >6342</td>
      <td id="T_ee5c2_row21_col14" class="data row21 col14" >7337</td>
      <td id="T_ee5c2_row21_col15" class="data row21 col15" >9148</td>
      <td id="T_ee5c2_row21_col16" class="data row21 col16" >10574</td>
      <td id="T_ee5c2_row21_col17" class="data row21 col17" >10962</td>
      <td id="T_ee5c2_row21_col18" class="data row21 col18" >9884</td>
      <td id="T_ee5c2_row21_col19" class="data row21 col19" >8980</td>
      <td id="T_ee5c2_row21_col20" class="data row21 col20" >8772</td>
      <td id="T_ee5c2_row21_col21" class="data row21 col21" >8430</td>
      <td id="T_ee5c2_row21_col22" class="data row21 col22" >6784</td>
      <td id="T_ee5c2_row21_col23" class="data row21 col23" >4530</td>
    </tr>
    <tr>
      <th id="T_ee5c2_level0_row22" class="row_heading level0 row22" >23</th>
      <td id="T_ee5c2_row22_col0" class="data row22 col0" >2546</td>
      <td id="T_ee5c2_row22_col1" class="data row22 col1" >1580</td>
      <td id="T_ee5c2_row22_col2" class="data row22 col2" >1136</td>
      <td id="T_ee5c2_row22_col3" class="data row22 col3" >1429</td>
      <td id="T_ee5c2_row22_col4" class="data row22 col4" >1957</td>
      <td id="T_ee5c2_row22_col5" class="data row22 col5" >3132</td>
      <td id="T_ee5c2_row22_col6" class="data row22 col6" >5204</td>
      <td id="T_ee5c2_row22_col7" class="data row22 col7" >6890</td>
      <td id="T_ee5c2_row22_col8" class="data row22 col8" >6436</td>
      <td id="T_ee5c2_row22_col9" class="data row22 col9" >5177</td>
      <td id="T_ee5c2_row22_col10" class="data row22 col10" >5066</td>
      <td id="T_ee5c2_row22_col11" class="data row22 col11" >5304</td>
      <td id="T_ee5c2_row22_col12" class="data row22 col12" >5504</td>
      <td id="T_ee5c2_row22_col13" class="data row22 col13" >6232</td>
      <td id="T_ee5c2_row22_col14" class="data row22 col14" >7575</td>
      <td id="T_ee5c2_row22_col15" class="data row22 col15" >9309</td>
      <td id="T_ee5c2_row22_col16" class="data row22 col16" >9980</td>
      <td id="T_ee5c2_row22_col17" class="data row22 col17" >10341</td>
      <td id="T_ee5c2_row22_col18" class="data row22 col18" >10823</td>
      <td id="T_ee5c2_row22_col19" class="data row22 col19" >11347</td>
      <td id="T_ee5c2_row22_col20" class="data row22 col20" >11447</td>
      <td id="T_ee5c2_row22_col21" class="data row22 col21" >10347</td>
      <td id="T_ee5c2_row22_col22" class="data row22 col22" >8637</td>
      <td id="T_ee5c2_row22_col23" class="data row22 col23" >5577</td>
    </tr>
    <tr>
      <th id="T_ee5c2_level0_row23" class="row_heading level0 row23" >24</th>
      <td id="T_ee5c2_row23_col0" class="data row23 col0" >3200</td>
      <td id="T_ee5c2_row23_col1" class="data row23 col1" >2055</td>
      <td id="T_ee5c2_row23_col2" class="data row23 col2" >1438</td>
      <td id="T_ee5c2_row23_col3" class="data row23 col3" >1493</td>
      <td id="T_ee5c2_row23_col4" class="data row23 col4" >1798</td>
      <td id="T_ee5c2_row23_col5" class="data row23 col5" >2754</td>
      <td id="T_ee5c2_row23_col6" class="data row23 col6" >4484</td>
      <td id="T_ee5c2_row23_col7" class="data row23 col7" >6013</td>
      <td id="T_ee5c2_row23_col8" class="data row23 col8" >5913</td>
      <td id="T_ee5c2_row23_col9" class="data row23 col9" >5146</td>
      <td id="T_ee5c2_row23_col10" class="data row23 col10" >4947</td>
      <td id="T_ee5c2_row23_col11" class="data row23 col11" >5311</td>
      <td id="T_ee5c2_row23_col12" class="data row23 col12" >5229</td>
      <td id="T_ee5c2_row23_col13" class="data row23 col13" >5974</td>
      <td id="T_ee5c2_row23_col14" class="data row23 col14" >7083</td>
      <td id="T_ee5c2_row23_col15" class="data row23 col15" >8706</td>
      <td id="T_ee5c2_row23_col16" class="data row23 col16" >10366</td>
      <td id="T_ee5c2_row23_col17" class="data row23 col17" >10786</td>
      <td id="T_ee5c2_row23_col18" class="data row23 col18" >9772</td>
      <td id="T_ee5c2_row23_col19" class="data row23 col19" >9080</td>
      <td id="T_ee5c2_row23_col20" class="data row23 col20" >9213</td>
      <td id="T_ee5c2_row23_col21" class="data row23 col21" >8831</td>
      <td id="T_ee5c2_row23_col22" class="data row23 col22" >7480</td>
      <td id="T_ee5c2_row23_col23" class="data row23 col23" >4456</td>
    </tr>
    <tr>
      <th id="T_ee5c2_level0_row24" class="row_heading level0 row24" >25</th>
      <td id="T_ee5c2_row24_col0" class="data row24 col0" >2405</td>
      <td id="T_ee5c2_row24_col1" class="data row24 col1" >1499</td>
      <td id="T_ee5c2_row24_col2" class="data row24 col2" >1072</td>
      <td id="T_ee5c2_row24_col3" class="data row24 col3" >1439</td>
      <td id="T_ee5c2_row24_col4" class="data row24 col4" >1943</td>
      <td id="T_ee5c2_row24_col5" class="data row24 col5" >2973</td>
      <td id="T_ee5c2_row24_col6" class="data row24 col6" >5356</td>
      <td id="T_ee5c2_row24_col7" class="data row24 col7" >7627</td>
      <td id="T_ee5c2_row24_col8" class="data row24 col8" >7078</td>
      <td id="T_ee5c2_row24_col9" class="data row24 col9" >5994</td>
      <td id="T_ee5c2_row24_col10" class="data row24 col10" >5432</td>
      <td id="T_ee5c2_row24_col11" class="data row24 col11" >5504</td>
      <td id="T_ee5c2_row24_col12" class="data row24 col12" >5694</td>
      <td id="T_ee5c2_row24_col13" class="data row24 col13" >6204</td>
      <td id="T_ee5c2_row24_col14" class="data row24 col14" >7298</td>
      <td id="T_ee5c2_row24_col15" class="data row24 col15" >8732</td>
      <td id="T_ee5c2_row24_col16" class="data row24 col16" >9922</td>
      <td id="T_ee5c2_row24_col17" class="data row24 col17" >10504</td>
      <td id="T_ee5c2_row24_col18" class="data row24 col18" >10673</td>
      <td id="T_ee5c2_row24_col19" class="data row24 col19" >9048</td>
      <td id="T_ee5c2_row24_col20" class="data row24 col20" >8751</td>
      <td id="T_ee5c2_row24_col21" class="data row24 col21" >9508</td>
      <td id="T_ee5c2_row24_col22" class="data row24 col22" >8522</td>
      <td id="T_ee5c2_row24_col23" class="data row24 col23" >6605</td>
    </tr>
    <tr>
      <th id="T_ee5c2_level0_row25" class="row_heading level0 row25" >26</th>
      <td id="T_ee5c2_row25_col0" class="data row25 col0" >3810</td>
      <td id="T_ee5c2_row25_col1" class="data row25 col1" >3065</td>
      <td id="T_ee5c2_row25_col2" class="data row25 col2" >2046</td>
      <td id="T_ee5c2_row25_col3" class="data row25 col3" >1806</td>
      <td id="T_ee5c2_row25_col4" class="data row25 col4" >1730</td>
      <td id="T_ee5c2_row25_col5" class="data row25 col5" >2337</td>
      <td id="T_ee5c2_row25_col6" class="data row25 col6" >3776</td>
      <td id="T_ee5c2_row25_col7" class="data row25 col7" >5172</td>
      <td id="T_ee5c2_row25_col8" class="data row25 col8" >5071</td>
      <td id="T_ee5c2_row25_col9" class="data row25 col9" >4808</td>
      <td id="T_ee5c2_row25_col10" class="data row25 col10" >5061</td>
      <td id="T_ee5c2_row25_col11" class="data row25 col11" >5179</td>
      <td id="T_ee5c2_row25_col12" class="data row25 col12" >5381</td>
      <td id="T_ee5c2_row25_col13" class="data row25 col13" >6166</td>
      <td id="T_ee5c2_row25_col14" class="data row25 col14" >7269</td>
      <td id="T_ee5c2_row25_col15" class="data row25 col15" >8815</td>
      <td id="T_ee5c2_row25_col16" class="data row25 col16" >9885</td>
      <td id="T_ee5c2_row25_col17" class="data row25 col17" >10697</td>
      <td id="T_ee5c2_row25_col18" class="data row25 col18" >10867</td>
      <td id="T_ee5c2_row25_col19" class="data row25 col19" >10122</td>
      <td id="T_ee5c2_row25_col20" class="data row25 col20" >9820</td>
      <td id="T_ee5c2_row25_col21" class="data row25 col21" >10441</td>
      <td id="T_ee5c2_row25_col22" class="data row25 col22" >9486</td>
      <td id="T_ee5c2_row25_col23" class="data row25 col23" >7593</td>
    </tr>
    <tr>
      <th id="T_ee5c2_level0_row26" class="row_heading level0 row26" >27</th>
      <td id="T_ee5c2_row26_col0" class="data row26 col0" >5196</td>
      <td id="T_ee5c2_row26_col1" class="data row26 col1" >3635</td>
      <td id="T_ee5c2_row26_col2" class="data row26 col2" >2352</td>
      <td id="T_ee5c2_row26_col3" class="data row26 col3" >2055</td>
      <td id="T_ee5c2_row26_col4" class="data row26 col4" >1723</td>
      <td id="T_ee5c2_row26_col5" class="data row26 col5" >2336</td>
      <td id="T_ee5c2_row26_col6" class="data row26 col6" >3539</td>
      <td id="T_ee5c2_row26_col7" class="data row26 col7" >4937</td>
      <td id="T_ee5c2_row26_col8" class="data row26 col8" >5053</td>
      <td id="T_ee5c2_row26_col9" class="data row26 col9" >4771</td>
      <td id="T_ee5c2_row26_col10" class="data row26 col10" >5198</td>
      <td id="T_ee5c2_row26_col11" class="data row26 col11" >5732</td>
      <td id="T_ee5c2_row26_col12" class="data row26 col12" >5839</td>
      <td id="T_ee5c2_row26_col13" class="data row26 col13" >6820</td>
      <td id="T_ee5c2_row26_col14" class="data row26 col14" >7519</td>
      <td id="T_ee5c2_row26_col15" class="data row26 col15" >8803</td>
      <td id="T_ee5c2_row26_col16" class="data row26 col16" >9793</td>
      <td id="T_ee5c2_row26_col17" class="data row26 col17" >9838</td>
      <td id="T_ee5c2_row26_col18" class="data row26 col18" >9228</td>
      <td id="T_ee5c2_row26_col19" class="data row26 col19" >8267</td>
      <td id="T_ee5c2_row26_col20" class="data row26 col20" >7908</td>
      <td id="T_ee5c2_row26_col21" class="data row26 col21" >8507</td>
      <td id="T_ee5c2_row26_col22" class="data row26 col22" >7720</td>
      <td id="T_ee5c2_row26_col23" class="data row26 col23" >6046</td>
    </tr>
    <tr>
      <th id="T_ee5c2_level0_row27" class="row_heading level0 row27" >28</th>
      <td id="T_ee5c2_row27_col0" class="data row27 col0" >4123</td>
      <td id="T_ee5c2_row27_col1" class="data row27 col1" >2646</td>
      <td id="T_ee5c2_row27_col2" class="data row27 col2" >1843</td>
      <td id="T_ee5c2_row27_col3" class="data row27 col3" >1802</td>
      <td id="T_ee5c2_row27_col4" class="data row27 col4" >1883</td>
      <td id="T_ee5c2_row27_col5" class="data row27 col5" >2793</td>
      <td id="T_ee5c2_row27_col6" class="data row27 col6" >4290</td>
      <td id="T_ee5c2_row27_col7" class="data row27 col7" >5715</td>
      <td id="T_ee5c2_row27_col8" class="data row27 col8" >5671</td>
      <td id="T_ee5c2_row27_col9" class="data row27 col9" >5206</td>
      <td id="T_ee5c2_row27_col10" class="data row27 col10" >5247</td>
      <td id="T_ee5c2_row27_col11" class="data row27 col11" >5500</td>
      <td id="T_ee5c2_row27_col12" class="data row27 col12" >5486</td>
      <td id="T_ee5c2_row27_col13" class="data row27 col13" >6120</td>
      <td id="T_ee5c2_row27_col14" class="data row27 col14" >7341</td>
      <td id="T_ee5c2_row27_col15" class="data row27 col15" >8584</td>
      <td id="T_ee5c2_row27_col16" class="data row27 col16" >9671</td>
      <td id="T_ee5c2_row27_col17" class="data row27 col17" >9975</td>
      <td id="T_ee5c2_row27_col18" class="data row27 col18" >9132</td>
      <td id="T_ee5c2_row27_col19" class="data row27 col19" >8255</td>
      <td id="T_ee5c2_row27_col20" class="data row27 col20" >8309</td>
      <td id="T_ee5c2_row27_col21" class="data row27 col21" >7949</td>
      <td id="T_ee5c2_row27_col22" class="data row27 col22" >6411</td>
      <td id="T_ee5c2_row27_col23" class="data row27 col23" >4461</td>
    </tr>
    <tr>
      <th id="T_ee5c2_level0_row28" class="row_heading level0 row28" >29</th>
      <td id="T_ee5c2_row28_col0" class="data row28 col0" >2678</td>
      <td id="T_ee5c2_row28_col1" class="data row28 col1" >1827</td>
      <td id="T_ee5c2_row28_col2" class="data row28 col2" >1409</td>
      <td id="T_ee5c2_row28_col3" class="data row28 col3" >1678</td>
      <td id="T_ee5c2_row28_col4" class="data row28 col4" >1948</td>
      <td id="T_ee5c2_row28_col5" class="data row28 col5" >3056</td>
      <td id="T_ee5c2_row28_col6" class="data row28 col6" >5213</td>
      <td id="T_ee5c2_row28_col7" class="data row28 col7" >6852</td>
      <td id="T_ee5c2_row28_col8" class="data row28 col8" >6695</td>
      <td id="T_ee5c2_row28_col9" class="data row28 col9" >5481</td>
      <td id="T_ee5c2_row28_col10" class="data row28 col10" >5234</td>
      <td id="T_ee5c2_row28_col11" class="data row28 col11" >5163</td>
      <td id="T_ee5c2_row28_col12" class="data row28 col12" >5220</td>
      <td id="T_ee5c2_row28_col13" class="data row28 col13" >6305</td>
      <td id="T_ee5c2_row28_col14" class="data row28 col14" >7630</td>
      <td id="T_ee5c2_row28_col15" class="data row28 col15" >9249</td>
      <td id="T_ee5c2_row28_col16" class="data row28 col16" >10105</td>
      <td id="T_ee5c2_row28_col17" class="data row28 col17" >11113</td>
      <td id="T_ee5c2_row28_col18" class="data row28 col18" >10411</td>
      <td id="T_ee5c2_row28_col19" class="data row28 col19" >9301</td>
      <td id="T_ee5c2_row28_col20" class="data row28 col20" >9270</td>
      <td id="T_ee5c2_row28_col21" class="data row28 col21" >9114</td>
      <td id="T_ee5c2_row28_col22" class="data row28 col22" >6992</td>
      <td id="T_ee5c2_row28_col23" class="data row28 col23" >4323</td>
    </tr>
    <tr>
      <th id="T_ee5c2_level0_row29" class="row_heading level0 row29" >30</th>
      <td id="T_ee5c2_row29_col0" class="data row29 col0" >2401</td>
      <td id="T_ee5c2_row29_col1" class="data row29 col1" >1510</td>
      <td id="T_ee5c2_row29_col2" class="data row29 col2" >1112</td>
      <td id="T_ee5c2_row29_col3" class="data row29 col3" >1403</td>
      <td id="T_ee5c2_row29_col4" class="data row29 col4" >1841</td>
      <td id="T_ee5c2_row29_col5" class="data row29 col5" >3216</td>
      <td id="T_ee5c2_row29_col6" class="data row29 col6" >5757</td>
      <td id="T_ee5c2_row29_col7" class="data row29 col7" >7596</td>
      <td id="T_ee5c2_row29_col8" class="data row29 col8" >7611</td>
      <td id="T_ee5c2_row29_col9" class="data row29 col9" >6064</td>
      <td id="T_ee5c2_row29_col10" class="data row29 col10" >5987</td>
      <td id="T_ee5c2_row29_col11" class="data row29 col11" >6090</td>
      <td id="T_ee5c2_row29_col12" class="data row29 col12" >6423</td>
      <td id="T_ee5c2_row29_col13" class="data row29 col13" >7249</td>
      <td id="T_ee5c2_row29_col14" class="data row29 col14" >8396</td>
      <td id="T_ee5c2_row29_col15" class="data row29 col15" >10243</td>
      <td id="T_ee5c2_row29_col16" class="data row29 col16" >11554</td>
      <td id="T_ee5c2_row29_col17" class="data row29 col17" >12126</td>
      <td id="T_ee5c2_row29_col18" class="data row29 col18" >12561</td>
      <td id="T_ee5c2_row29_col19" class="data row29 col19" >11024</td>
      <td id="T_ee5c2_row29_col20" class="data row29 col20" >10836</td>
      <td id="T_ee5c2_row29_col21" class="data row29 col21" >10042</td>
      <td id="T_ee5c2_row29_col22" class="data row29 col22" >8275</td>
      <td id="T_ee5c2_row29_col23" class="data row29 col23" >4723</td>
    </tr>
    <tr>
      <th id="T_ee5c2_level0_row30" class="row_heading level0 row30" >31</th>
      <td id="T_ee5c2_row30_col0" class="data row30 col0" >2174</td>
      <td id="T_ee5c2_row30_col1" class="data row30 col1" >1394</td>
      <td id="T_ee5c2_row30_col2" class="data row30 col2" >1087</td>
      <td id="T_ee5c2_row30_col3" class="data row30 col3" >919</td>
      <td id="T_ee5c2_row30_col4" class="data row30 col4" >773</td>
      <td id="T_ee5c2_row30_col5" class="data row30 col5" >997</td>
      <td id="T_ee5c2_row30_col6" class="data row30 col6" >1561</td>
      <td id="T_ee5c2_row30_col7" class="data row30 col7" >2169</td>
      <td id="T_ee5c2_row30_col8" class="data row30 col8" >2410</td>
      <td id="T_ee5c2_row30_col9" class="data row30 col9" >2525</td>
      <td id="T_ee5c2_row30_col10" class="data row30 col10" >2564</td>
      <td id="T_ee5c2_row30_col11" class="data row30 col11" >2777</td>
      <td id="T_ee5c2_row30_col12" class="data row30 col12" >2954</td>
      <td id="T_ee5c2_row30_col13" class="data row30 col13" >3280</td>
      <td id="T_ee5c2_row30_col14" class="data row30 col14" >4104</td>
      <td id="T_ee5c2_row30_col15" class="data row30 col15" >5099</td>
      <td id="T_ee5c2_row30_col16" class="data row30 col16" >5386</td>
      <td id="T_ee5c2_row30_col17" class="data row30 col17" >5308</td>
      <td id="T_ee5c2_row30_col18" class="data row30 col18" >5350</td>
      <td id="T_ee5c2_row30_col19" class="data row30 col19" >4898</td>
      <td id="T_ee5c2_row30_col20" class="data row30 col20" >4819</td>
      <td id="T_ee5c2_row30_col21" class="data row30 col21" >5064</td>
      <td id="T_ee5c2_row30_col22" class="data row30 col22" >5164</td>
      <td id="T_ee5c2_row30_col23" class="data row30 col23" >3961</td>
    </tr>
  </tbody>
</table>



