#!/usr/bin/env python
# coding: utf-8

# # SHAP with structured data classification

# In[1]:


import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import shap

tf.__version__


# In[2]:


# print the JS visualization code to the notebook
shap.initjs()


# In[3]:


df = pd.read_csv('car_prices.csv',error_bad_lines=False,warn_bad_lines=True)
print(df) # Ausgabe 


# In[4]:


df.info()


# In[ ]:





# In[5]:


# make target variable
y = df.pop('sellingprice')


# In[12]:


# prepare features
list_numerical = ['year', 'mmr']

X = df[list_numerical]


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[14]:


scaler = StandardScaler().fit(X_train[list_numerical]) 

X_train[list_numerical] = scaler.transform(X_train[list_numerical])
X_test[list_numerical] = scaler.transform(X_test[list_numerical])


# In[15]:


model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
  ])


# In[16]:


model.compile(optimizer="adam", 
              loss ="binary_crossentropy", 
              metrics=["accuracy"])


# In[17]:


model.fit(X_train, y_train, 
         epochs=15, 
         batch_size=13,
         validation_data=(X_test, y_test)
         )


# In[18]:


# `rankdir='LR'` is to make the graph horizontal.
tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")


# In[19]:


loss, accuracy = model.evaluate(X_test, y_test)

print("Accuracy", accuracy)


# In[20]:


model.save('classifier_hd')


# In[21]:


reloaded_model = tf.keras.models.load_model('classifier_hd')


# In[22]:


predictions = reloaded_model.predict(X_train)


# In[ ]:


print(
    "This particular patient had a %.1f percent probability "
    "of having a heart disease, as evaluated by our model." % (100 * predictions[0][0],)
)


# In[23]:


explainer = shap.KernelExplainer(model, X_train.iloc[:50,:])


# In[24]:


shap_values = explainer.shap_values(X_train.iloc[20,:], nsamples=500)


# In[25]:


shap.force_plot(explainer.expected_value, shap_values[0], X_train.iloc[20,:])


# In[28]:


shap_values50 = explainer.shap_values(X_train.iloc[50:100,:], nsamples=500)


# In[29]:


shap.force_plot(explainer.expected_value, shap_values50[0], X_train.iloc[50:100,:])


# In[ ]:





# 
