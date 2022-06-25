#!/usr/bin/env python
# coding: utf-8

# # SHAP mit strukturierten Daten (Classification)

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


df = pd.read_csv('Mobile_Price_train.csv',error_bad_lines=False,warn_bad_lines=True)
print(df) # Ausgabe 


# In[4]:


df.info()


# In[ ]:





# In[5]:


# make target variable
y = df.pop('price_range')


# In[6]:


# prepare features
list_numerical = ['ram', 'battery_power', 'touch_screen']

X = df[list_numerical]


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[8]:


scaler = StandardScaler().fit(X_train[list_numerical]) 

X_train[list_numerical] = scaler.transform(X_train[list_numerical])
X_test[list_numerical] = scaler.transform(X_test[list_numerical])


# In[9]:


model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
  ])


# In[10]:


model.compile(optimizer="adam", 
              loss ="binary_crossentropy", 
              metrics=["accuracy"])


# In[11]:


model.fit(X_train, y_train, 
         epochs=15, 
         batch_size=13,
         validation_data=(X_test, y_test)
         )


# In[12]:


# `rankdir='LR'` is to make the graph horizontal.
tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")


# In[13]:


loss, accuracy = model.evaluate(X_test, y_test)

print("Accuracy", accuracy)


# In[14]:


model.save('classifier_mobile_hd')


# In[15]:


reloaded_model = tf.keras.models.load_model('classifier_mobile_hd')


# In[16]:


predictions = reloaded_model.predict(X_train)


# In[17]:


print(
    "This particular patient had a %.1f percent probability "
    "of having a heart disease, as evaluated by our model." % (100 * predictions[0][0],)
)


# In[18]:


explainer = shap.KernelExplainer(model, X_train.iloc[:50,:])


# In[19]:


shap_values = explainer.shap_values(X_train.iloc[20,:], nsamples=500)


# In[20]:


shap.force_plot(explainer.expected_value, shap_values[0], X_train.iloc[20,:])


# In[21]:


shap_values50 = explainer.shap_values(X_train.iloc[50:100,:], nsamples=500)


# In[22]:


shap.force_plot(explainer.expected_value, shap_values50[0], X_train.iloc[50:100,:])


# In[ ]:





# 
