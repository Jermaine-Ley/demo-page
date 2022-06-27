#!/usr/bin/env python
# coding: utf-8

# # Klassifikation (Mobile Dataset)

# In[1]:


import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers

tf.__version__


# In[2]:


df = pd.read_csv('Mobile_Price_train.csv',error_bad_lines=False,warn_bad_lines=True)
print(df) # Ausgabe 


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


# To find the number of duplicate rows

duplicate_rows_df = df[df.duplicated()]
print("number of duplicate rows: ", duplicate_rows_df.shape)


# In[6]:


#Fehlende Werte erkennen. Gibt ein boolesches Objekt zurück, das angibt, ob die Werte NA sind 
df.isna().sum().sort_values(ascending=False)


# In[7]:


#  To Drop the missing or null values
print(df.isnull().sum())     # Finding the number of Null values


# In[8]:


df = df.dropna()    # Dropping the missing values.
df.count()


# In[9]:


print(df.isnull().sum())   # After dropping the values


# In[10]:


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


# In[11]:


y_label = 'price_range'


# In[12]:


# Make a dictionary with int64 featureumns as keys and np.int32 as values
int_32 = dict.fromkeys(df.select_dtypes(np.int64).columns, np.int32)
# Change all columns from dictionary
df = df.astype(int_32)

# Make a dictionary with float64 columns as keys and np.float32 as values
float_32 = dict.fromkeys(df.select_dtypes(np.float64).columns, np.float32)
df = df.astype(float_32)


# In[13]:


int_32


# In[14]:


# Convert to numeric

# make a list of all categorical variables
cat_convert = ['ram', 'battery_power', 'touch_screen']

# convert variables
for i in cat_convert:
    df[i] = df[i].astype("int")


# In[15]:


# Make list of all numerical data (except label)
list_num = df.drop(columns=[y_label]).select_dtypes(include=[np.number]).columns.tolist()

# Make list of all categorical data which is stored as integers (except label)
list_cat_int = df.drop(columns=[y_label]).select_dtypes(include=['category']).columns.tolist()

# Make list of all categorical data which is stored as string (except label)
list_cat_string = df.drop(columns=[y_label]).select_dtypes(include=['string']).columns.tolist()


# In[16]:


df.info()


# In[17]:


# Make validation data
df_val = df.sample(frac=0.2, random_state=1337)

# Create training data
df_train_mobile_class = df.drop(df_val.index)


# In[18]:


# Save training data
df_train_mobile_class.to_csv("df_train_mobile_class.csv", index=False)


# In[19]:


print(
    "Using %d samples for training and %d for validation"
    % (len(df_train_mobile_class), len(df_val))
)


# In[20]:


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


# In[21]:


batch_size = 32

ds_train = dataframe_to_dataset(df_train_mobile_class, shuffle=True, batch_size=batch_size)
ds_val = dataframe_to_dataset(df_val, shuffle=True, batch_size=batch_size)


# In[22]:


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


# In[23]:


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


# In[24]:


all_inputs = []
encoded_features = []


# In[25]:


# Numerical features
for feature in list_num:
  numeric_feature = tf.keras.Input(shape=(1,), name=feature)
  normalization_layer = get_normalization_layer(feature, ds_train)
  encoded_numeric_feature = normalization_layer(numeric_feature)
  all_inputs.append(numeric_feature)
  encoded_features.append(encoded_numeric_feature)


# In[26]:


for feature in list_cat_int:
  categorical_feature = tf.keras.Input(shape=(1,), name=feature, dtype='int32')
  encoding_layer = get_category_encoding_layer(name=feature,
                                               dataset=ds_train,
                                               dtype='int32',
                                               max_tokens=5)
  encoded_categorical_feature = encoding_layer(categorical_feature)
  all_inputs.append(categorical_feature)
  encoded_features.append(encoded_categorical_feature)


# In[27]:


for feature in list_cat_string:
  categorical_feature = tf.keras.Input(shape=(1,), name=feature, dtype='string')
  encoding_layer = get_category_encoding_layer(name=feature,
                                               dataset=ds_train,
                                               dtype='string',
                                               max_tokens=5)
  encoded_categorical_feature = encoding_layer(categorical_feature)
  all_inputs.append(categorical_feature)
  encoded_features.append(encoded_categorical_feature)


# In[28]:


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


# In[29]:


model.compile(optimizer="adam", 
              loss ="binary_crossentropy", 
              metrics=["accuracy"])


# In[30]:


# `rankdir='LR'` is to make the graph horizontal.
tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")


# Training

# In[31]:


loss, accuracy = model.evaluate(ds_val)

print("Accuracy", round(accuracy, 2))


# In[32]:


model.save('my_hd_classifier_mobile_classi')


# In[33]:


reloaded_model = tf.keras.models.load_model('my_hd_classifier_mobile_classi')


# In[49]:


sample = {
'battery_power': 1500,
'px_height': 1,  
'wifi': 1,
'touch_screen': 1,    
'three_g': 1,
'talk_time': 100,
'sc_w': 8,
'sc_h': 12,
'ram': 4873,
'px_width': 957,
'pc': 20,
'blue': 1,
'n_cores': 5, 
'mobile_wt': 156,
'm_dep': 0.8,
'int_memory':50, 
'four_g': 1,
'fc': 14,
'dual_sim': 1,
'clock_speed': 0.5,
'price_range': "fixed",
}


# In[50]:


input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}


# In[51]:


predictions = reloaded_model.predict(input_dict)


# In[52]:


print(
    "Mit diesen ausgewählten Parametern besteht die Auswahl für ein gutes Smartphone bei einer %.1f prozentigen Wahrscheinlichkeit. "
     % (100 * predictions[0][0],)
)

