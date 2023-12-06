#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

data = pd.read_csv('2019_kbo_for_kaggle_v2.csv')
data


# In[6]:


year = data['year'].values
range = (year >= 2015) & (year <= 2018)

for item in ['H', 'avg', 'HR', 'OBP']:
    temp1 = data[['year', item]]

    print(temp1[range].sort_values(by=item, ascending=False)[:10])


# In[47]:


sample = data[['cp', 'year', 'batter_name', 'war']]

for item in ['포수', '1루수', '2루수', '3루수', '유격수', '좌익수', '중견수', '우익수']:
    newData = sample.loc[(sample['year'] == 2018) & (sample['cp'] == item)]
    print(newData.sort_values(by='war', ascending=False)[:1])


# In[80]:


sample = data[['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG']]
newData = sample.corrwith(data.salary)
newData = newData.to_frame()
newData.columns = ['correlation']
newData.sort_values('correlation', ascending=False).iloc[:1]


# In[ ]:




