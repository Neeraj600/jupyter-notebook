#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# In[20]:


iris = pd.read_csv('C:/Users/Ajay Guleria/Desktop/iris.csv')


# In[21]:


iris.head()


# In[22]:


iris[iris['SepalWidthCm']>4]


# In[23]:


iris[iris['PetalWidthCm']>1]


# In[24]:


iris[iris['PetalWidthCm']>2]


# In[28]:


sns.scatterplot(x='SepalLengthCm',y='PetalLengthCm',data=iris,hue='Species')


# In[29]:


iris.head()


# In[52]:


y = iris [['SepalLengthCm']]


# In[53]:


x = iris[['SepalWidthCm']]


# In[54]:


from sklearn.model_selection import train_test_split


# In[55]:


#test_size means 30% of the record will be in test set and 70% of the record will be in train record 
#x_train will be a training set of independent variable
#x_test will be a testinging set of independent variable
#y_train will be a training set of dependent variable
#y_test will be a testinging set of dependent variable(sequence matters)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[56]:


x_train.head()


# In[57]:


x_test.head()


# In[58]:


y_train.head()


# In[59]:


y_test.head()


# In[65]:


from sklearn.linear_model import LinearRegression


# In[66]:


lr = LinearRegression()


# In[67]:


lr.fit(x_train, y_train)


# In[69]:


#predicting test set
y_pred=lr.predict(x_test)


# In[70]:


y_test.head()


# In[74]:


y_pred[0:5]


# In[76]:


from sklearn.metrics import mean_squared_error


# In[77]:


mean_squared_error(y_test,y_pred)


# In[78]:


#model 2


# In[80]:


y = iris[['SepalLengthCm']]
   


# In[84]:


x = iris[['SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']] 


# In[86]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[89]:


lr2 = LinearRegression()


# In[91]:


lr2.fit(x_train, y_train)


# In[95]:


y_pred=lr2.predict(x_test)


# In[96]:


mean_squared_error(y_test, y_pred)


# In[ ]:





# In[97]:


#covid 19 data set


# In[270]:


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import datetime as dt
import numpy as np


# In[271]:


df = pd.read_csv('C:/Users/Ajay Guleria/Desktop/data sets/covid_19_india.csv', parse_dates=['Date'], dayfirst=True)


# In[272]:


df.head()


# In[273]:


#keeping only required columns
df = df[['Date','State/UnionTerritory', 'Cured', 'Deaths', 'Confirmed']]
#renaming columns
df.columns = ['date', 'state' , 'cured', 'deaths', 'confirmed']


# In[274]:


df.head()


# In[275]:


df.tail()


# In[276]:


today = df[df.date=='2020-06-22']


# In[277]:


today.head()


# In[278]:


max_confirmed_cases = today.sort_values(by='confirmed', ascending=False)
max_confirmed_cases


# In[279]:


top_states_confirmed = max_confirmed_cases[0:5]
top_states_confirmed


# In[280]:


sns.set(rc={'figure.figsize':(8,8)})
sns.barplot(x='state', y='confirmed',data=top_states_confirmed, hue='state')


# In[304]:


#sorting data wrt deaths
max_death_cases = today.sort_values(by='deaths',ascending=False)
max_death_cases


# In[305]:


top_deaths_states = max_death_cases[:5]
top_deaths_states


# In[306]:


sns.set(rc={'figure.figsize':(8,8)})
sns.barplot(x='state', y= 'deaths', data=top_deaths_states, hue='state')


# In[307]:


max_cured_cases = today.sort_values(by='cured', ascending=False)
max_cured_cases 


# In[308]:


top_cured_states = max_cured_cases [:5]
top_cured_states


# In[309]:


sns.set(rc={'figure.figsize':(8,8)})
sns.barplot(x='state', y='cured', data=top_cured_states, hue='cured')


# In[310]:


maha = df[df.state== 'Maharashtra']


# In[311]:


maha


# In[312]:


sns.set(rc={'figure.figsize':(8,8)})
sns.lineplot(x='date', y= 'confirmed', data=maha, color='g')


# In[313]:


sns.set(rc={'figure.figsize':(8,8)})
sns.lineplot(x='date', y= 'deaths', data=maha, color='g')


# In[314]:


Kerala = df[df.state =='Kerala']
Kerala


# In[315]:


sns.set(rc={'figure.figsize':(8,8)})
sns.lineplot(x='date', y= 'confirmed', data=Kerala, color='g')


# In[316]:


sns.set(rc={'figure.figsize':(8,8)})
sns.lineplot(x='date', y= 'deaths', data=Kerala, color='g')


# In[317]:


jk = df[df.state=='Jammu and Kashmir']
jk


# In[318]:


sns.set(rc={'figure.figsize':(8,8)})
sns.lineplot(x='date', y= 'confirmed', data=jk, color='g')


# In[319]:


sns.set(rc={'figure.figsize':(8,8)})
sns.lineplot(x='date', y= 'deaths', data=jk, color='g')


# In[320]:


from sklearn.model_selection import train_test_split


# In[321]:


maha


# In[299]:


#convrting date and time to ordinal(in order)
maha['date']=maha['date'].map(dt.datetime.toordinal) 
maha.head()


# In[ ]:




x=maha['date']
y=maha['confirmed']
# In[322]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[323]:


from sklearn.linear_model import LinearRegression


# In[324]:


lr = LinearRegression()


# In[325]:


y_train


# In[326]:


lr.fit(np.array(x_train).reshape(-1,1),np.array(x_train).reshape(-1,1))


# In[327]:


maha.tail()


# In[331]:


lr.predict(np.array([[2020-6-16]]))


# In[ ]:





# In[332]:


#logistic regression


# In[2]:


import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
color = sns.color_palette
import sklearn.metrics as metrics 
import warnings
warnings.filterwarnings('ignore')


# In[3]:


default = pd.read_csv('C:/Users/Ajay Guleria/Desktop/data sets/default.csv')


# In[4]:


default.head()


# In[5]:


default.shape


# In[7]:


default.describe()


# In[10]:


#small dots are outliers
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.boxplot(y=default['balance'])

plt.subplot(1,2,2)
sns.boxplot(y=default['income'])


# In[13]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.countplot(y=default['student'])

plt.subplot(1,2,2)
sns.countplot(y=default)['default']
plt.show()


# In[ ]:




