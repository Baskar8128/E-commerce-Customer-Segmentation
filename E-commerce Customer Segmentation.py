#!/usr/bin/env python
# coding: utf-8

# In[48]:


# Importing libraries
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


# In[2]:


# reading the data
df = pd.read_excel('cust_data.xlsx')
df.head()


# In[3]:


df.shape


# In[4]:


# Distribution of orders
sns.countplot(data=df, x = 'Orders')
plt.title('Distribution of Orders', fontsize = 15)
plt.xlabel('NO of Orders', fontsize = 15)
plt.ylabel('Count', fontsize = 15)


# In[5]:


sns.countplot(data=df, x = 'Gender')
plt.title('Distribution of Gender', fontsize = 15)
plt.xlabel('Gender', fontsize = 15)
plt.ylabel('No of Customers', fontsize = 15)


# In[6]:


# describing the data
df.describe()


# In[7]:


# heatmap for missing values
sns.heatmap(df.isnull(),cbar = False)
plt.title('Heatmap of missing data', fontsize = 15)
plt.xlabel('Variable', fontsize = 15)
plt.ylabel('Cust_ID', fontsize = 15)
plt.show()


# In[8]:


# Checking for the column value type
df.info()


# In[39]:


# creating dummies from Gender column
df1 = df.drop(['Cust_ID'],axis=1)
df1 = pd.get_dummies(df1, columns=['Gender'])
df1.head()


# In[34]:


df1.shape


# In[12]:


# Data Visualizzation(histogram)
fig = df1.hist(figsize=(25,25))


# In[40]:


# dropping unwanted columns
features = df1.drop(['Orders','Gender_M', 'Gender_F'],axis=1)
features.head()


# In[41]:


# Normalizing the data
from sklearn.preprocessing import StandardScaler
scale = StandardScaler().fit(features)
#Scale the features data
features = scale.transform(features)


# In[42]:


feature_df = pd.DataFrame(features, columns = df1.columns[1:36])
feature_df.head()


# In[43]:


# finding the best cluster and silhouette score
import warnings
warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
n_clusters = [4, 5, 6, 7]
for k in n_clusters:
    cluster = KMeans(n_clusters = k, random_state=10)
    predict = cluster.fit_predict(feature_df)
    score = silhouette_score(feature_df, predict, random_state=10)
    print("For Clus = {}, silhouette_score is {})".format(k, score))


# In[44]:


model = KMeans(n_clusters=4, random_state=10)
model.fit(feature_df)


# In[46]:


df_output = df1.copy(deep=True)

# inserting cluster column 
df_output['Cluster']= model.labels_
df_output.head()


# In[49]:


#checking the size of the cluster
np.unique(model.labels_, return_counts = True)


# In[50]:


#discribing cluster size using seaborn barplot
sns.countplot(data= df_output, x = 'Cluster')
plt.title('Cluster sizes', fontsize = 15)
plt.xlabel('Clusters', fontsize = 15)
plt.ylabel('No of Customers', fontsize = 15)


# In[ ]:




