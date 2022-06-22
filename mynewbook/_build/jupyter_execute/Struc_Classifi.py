#!/usr/bin/env python
# coding: utf-8

# # Classification

# In[34]:


import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers

tf.__version__


# In[35]:


df = pd.read_csv('car_prices.csv',error_bad_lines=False,warn_bad_lines=True)
print(df) # Ausgabe 


# In[36]:


df.head()


# In[37]:


df.info()


# In[38]:


# To find the number of duplicate rows

duplicate_rows_df = df[df.duplicated()]
print("number of duplicate rows: ", duplicate_rows_df.shape)


# In[39]:


#Fehlende Werte erkennen. Gibt ein boolesches Objekt zurück, das angibt, ob die Werte NA sind 
df.isna().sum().sort_values(ascending=False)


# In[40]:


#  To Drop the missing or null values
print(df.isnull().sum())     # Finding the number of Null values


# In[41]:


df = df.dropna()    # Dropping the missing values.
df.count()


# In[42]:


print(df.isnull().sum())   # After dropping the values


# In[43]:


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


# In[44]:


categorical_columns


# Define label

# In[45]:


y_label = 'sellingprice'


# In[46]:


# Make a dictionary with int64 featureumns as keys and np.int32 as values
int_32 = dict.fromkeys(df.select_dtypes(np.int64).columns, np.int32)
# Change all columns from dictionary
df = df.astype(int_32)

# Make a dictionary with float64 columns as keys and np.float32 as values
float_32 = dict.fromkeys(df.select_dtypes(np.float64).columns, np.float32)
df = df.astype(float_32)


# In[47]:


int_32


# In[48]:


# Convert to numeric

# make a list of all categorical variables
cat_convert = ['year', 'condition', 'odometer', 'mmr']

# convert variables
for i in cat_convert:
    df[i] = df[i].astype("int")


# In[ ]:





# In[49]:


# Make list of all numerical data (except label)
list_num = df.drop(columns=[y_label]).select_dtypes(include=[np.number]).columns.tolist()

# Make list of all categorical data which is stored as integers (except label)
list_cat_int = df.drop(columns=[y_label]).select_dtypes(include=['category']).columns.tolist()

# Make list of all categorical data which is stored as string (except label)
list_cat_string = df.drop(columns=[y_label]).select_dtypes(include=['string']).columns.tolist()


# In[50]:


df.info()


# Data splitting

# In[51]:


# Make validation data
df_val = df.sample(frac=0.2, random_state=1337)

# Create training data
df_train = df.drop(df_val.index)


# In[52]:


# Save training data
df_train.to_csv("df_train.csv", index=False)


# In[53]:


print(
    "Using %d samples for training and %d for validation"
    % (len(df_train), len(df_val))
)


# Transform to Tensors

# In[54]:


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


# In[55]:


batch_size = 32

ds_train = dataframe_to_dataset(df_train, shuffle=True, batch_size=batch_size)
ds_val = dataframe_to_dataset(df_val, shuffle=True, batch_size=batch_size)


# Numerical preprocessing function

# In[56]:


# Define numerical preprocessing function
def get_normalization_layer(name, dataset):
    
    # Create a Normalization layer for our feature
    normalizer = layers.Normalization(axis=None)

    # Prepare a dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])

    # Learn the statistics of the data
    normalizer.adapt(feature_ds)

    # Normalize the input feature
    return normalizer


# Categorical preprocessing functions

# In[57]:


def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
  
  # Create a layer that turns strings into integer indices.
  if dtype == 'string':
    index = layers.StringLookup(max_tokens=max_tokens)
  # Otherwise, create a layer that turns integer values into integer indices.
  else:
    index = layers.IntegerLookup(max_tokens=max_tokens)

  # Prepare a `tf.data.Dataset` that only yields the feature.
  feature_ds = dataset.map(lambda x, y: x[name])

  # Learn the set of possible values and assign them a fixed integer index.
  index.adapt(feature_ds)

  # Encode the integer indices.
  encoder = layers.CategoryEncoding(num_tokens=index.vocabulary_size())

  # Apply multi-hot encoding to the indices. The lambda function captures the
  # layer, so you can use them, or include them in the Keras Functional model later.
  return lambda feature: encoder(index(feature))


# Data preprocessing

# In[58]:


all_inputs = []
encoded_features = []


# Numerical preprocessing

# In[59]:


# Numerical features
for feature in list_num:
  numeric_feature = tf.keras.Input(shape=(1,), name=feature)
  normalization_layer = get_normalization_layer(feature, ds_train)
  encoded_numeric_feature = normalization_layer(numeric_feature)
  all_inputs.append(numeric_feature)
  encoded_features.append(encoded_numeric_feature)


# Categorical preprocessing

# In[60]:


for feature in list_cat_int:
  categorical_feature = tf.keras.Input(shape=(1,), name=feature, dtype='int32')
  encoding_layer = get_category_encoding_layer(name=feature,
                                               dataset=ds_train,
                                               dtype='int32',
                                               max_tokens=5)
  encoded_categorical_feature = encoding_layer(categorical_feature)
  all_inputs.append(categorical_feature)
  encoded_features.append(encoded_categorical_feature)


# In[61]:


for feature in list_cat_string:
  categorical_feature = tf.keras.Input(shape=(1,), name=feature, dtype='string')
  encoding_layer = get_category_encoding_layer(name=feature,
                                               dataset=ds_train,
                                               dtype='string',
                                               max_tokens=5)
  encoded_categorical_feature = encoding_layer(categorical_feature)
  all_inputs.append(categorical_feature)
  encoded_features.append(encoded_categorical_feature)


# Model

# In[62]:


# Input
all_features = layers.concatenate(encoded_features)

# First layer
x = layers.Dense(32, activation="relu")(all_features)

# Dropout to prevent overvitting
x = layers.Dropout(0.5)(x)

# Output layer
output = layers.Dense(1, activation="sigmoid")(x)

# Group all layers 
model = tf.keras.Model(all_inputs, output)


# In[ ]:


model.compile(optimizer='adam',loss='mean_squared_error')


# In[63]:


# model.compile(optimizer="adam", 
#               loss ="binary_crossentropy", 
#               metrics=["accuracy"])


# In[64]:


# `rankdir='LR'` is to make the graph horizontal.
tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")


# In[65]:


df.info()


# In[66]:


get_ipython().run_cell_magic('time', '', 'model.fit(ds_train, epochs=10, validation_data=ds_val)\n')


# In[ ]:


loss, mean_squared_error = model.evaluate(ds_val)

print("mean_squared_error", round(mean_squared_error, 2))


# In[67]:


loss, accuracy = model.evaluate(ds_val)

print("Accuracy", round(accuracy, 2))


# Perform inference

# In[68]:


model.save('my_hd_classifier')


# In[69]:


reloaded_model = tf.keras.models.load_model('my_hd_classifier')


# In[70]:


sample = {
    "year": 2017,
    "make": "BMW",
    "model": "3 Series",
    "trim": "328i SULEV",
    "body": "sedan",
    "transmission": "automatic",
    "vin": "wba3c1c51ek116351",
    "state": "ca",
    "condition": 5.0,
    "odometer": 1300
}

