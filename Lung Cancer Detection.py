#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
ds=pd.read_csv("survey lung cancer.csv")


# In[2]:


ds.head()


# In[3]:


ds.tail()


# In[4]:


ds.shape


# In[5]:


ds.isnull().sum()


# In[6]:


ds.duplicated().sum()


# In[7]:


ds.describe()


# In[8]:


ds.info()


# In[9]:


ds = ds.drop_duplicates()


# In[10]:


ds.duplicated().sum()


# In[11]:


print(ds)


# In[12]:


ds['GENDER'].unique()


# In[13]:


len(ds.columns)


# In[14]:


from sklearn.preprocessing import LabelEncoder
enc=LabelEncoder()


# In[15]:


ds['GENDER']=enc.fit_transform(ds['GENDER'])


# In[16]:


ds


# In[17]:


from sklearn.linear_model import LogisticRegression


# In[18]:


ds['LUNG_CANCER'].unique()


# In[19]:


data={'YES':1,'NO':0}
ds['LUNG_CANCER']=ds['LUNG_CANCER'].map(data)


# In[20]:


ds.head()


# In[21]:


x=ds.drop('LUNG_CANCER',axis=1)


# In[22]:


x.head()


# In[23]:


y=ds['LUNG_CANCER']


# In[24]:


ds


# In[25]:


y.head()


# In[26]:


from sklearn.model_selection import train_test_split


# In[27]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[28]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[29]:


model.fit(x_train, y_train)


# In[30]:


model.predict(x_test)


# In[31]:


import pickle as pkl


# In[32]:


pkl.dump(model,open('abc.pkl','wb'))


# In[ ]:




