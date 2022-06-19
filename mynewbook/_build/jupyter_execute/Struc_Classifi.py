#!/usr/bin/env python
# coding: utf-8

# # Classification

# In[27]:


import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers

tf.__version__


# In[28]:


df = pd.read_csv('car_prices.csv',error_bad_lines=False,warn_bad_lines=True)
print(df) # Ausgabe 


# In[29]:


df.head()


# In[30]:


df.info()


# In[43]:


#Fehlende Werte erkennen. Gibt ein boolesches Objekt zurück, das angibt, ob die Werte NA sind 
df.isna().sum().sort_values(ascending=False)


# In[45]:


categorical_columns = []
continous_columns = []
discrete_columns = []

for x in df.columns:
  if df[x].dtypes == 'O':
    categorical_columns.append(x)
  else:
    if df[x].nunique()>20:
      continous_columns.append(x)
    else:
      discrete_columns.append(x)


# In[46]:


categorical_columns


# In[ ]:





# In[31]:


y_label = 'make'


# In[32]:


# Make a dictionary with int64 featureumns as keys and np.int32 as values
int_32 = dict.fromkeys(df.select_dtypes(np.int64).columns, np.int32)
# Change all columns from dictionary
df = df.astype(int_32)

# Make a dictionary with float64 columns as keys and np.float32 as values
float_32 = dict.fromkeys(df.select_dtypes(np.float64).columns, np.float32)
df = df.astype(float_32)


# In[33]:


int_32


# In[42]:


# Convert to categorical

# make a list of all categorical variables
cat_convert = ['odometer', 'condition']

# convert variables
for i in cat_convert:
    df[i] = df[i].astype("int")


# In[ ]:





# In[34]:


# Make list of all numerical data (except label)
list_num = df.drop(columns=[y_label]).select_dtypes(include=[np.number]).columns.tolist()

# Make list of all categorical data which is stored as integers (except label)
list_cat_int = df.drop(columns=[y_label]).select_dtypes(include=['category']).columns.tolist()

# Make list of all categorical data which is stored as string (except label)
list_cat_string = df.drop(columns=[y_label]).select_dtypes(include=['string']).columns.tolist()


# In[35]:


df.info()


# Data splitting

# In[36]:


# Make validation data
df_val = df.sample(frac=0.2, random_state=1337)

# Create training data
df_train = df.drop(df_val.index)


# In[37]:


# Save training data
df_train.to_csv("df_train.csv", index=False)


# In[38]:


print(
    "Using %d samples for training and %d for validation"
    % (len(df_train), len(df_val))
)


# Transform to Tensors

# In[39]:


# Define a function to create our tensors

def dataframe_to_dataset(dataframe, shuffle=True, batch_size=32):
    df = dataframe.copy()
    labels = df.pop(y_label)
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    df = ds.prefetch(batch_size)
    return ds


# In[41]:


batch_size = 32

ds_train = dataframe_to_dataset(df_train, shuffle=True, batch_size=batch_size)
ds_val = dataframe_to_dataset(df_val, shuffle=True, batch_size=batch_size)

