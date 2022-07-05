#!/usr/bin/env python
# coding: utf-8

# # SHAP mit strukturierten Daten (Classification) Mobile Dataset

# Anlehnung an Classification I & Classification II von Prof. Dr. Jan Kirenz
# <br>
# https://kirenz.github.io/deep-learning/docs/shap_structured_data_classification.html

# In[21]:


import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import shap

tf.__version__


# In[22]:


# print the JS visualization code to the notebook
shap.initjs()


# In[23]:


df = pd.read_csv('Mobile_Price_train.csv',error_bad_lines=False,warn_bad_lines=True)
print(df) # Ausgabe 


# In[24]:


df.info()


# In[25]:


# make target variable
y = df.pop('price_range')


# In[26]:


# prepare features
list_numerical = ['ram', 'battery_power', 'touch_screen', 'int_memory', 'pc', 'blue','clock_speed', 'dual_sim', 'fc','four_g','m_dep', 'mobile_wt', 'n_cores', 'px_height', 'px_width', 'sc_h', 'sc_w', 'talk_time', 'three_g', 'wifi']

X = df[list_numerical]


# In[27]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[28]:


scaler = StandardScaler().fit(X_train[list_numerical]) 

X_train[list_numerical] = scaler.transform(X_train[list_numerical])
X_test[list_numerical] = scaler.transform(X_test[list_numerical])


# In[29]:


model = tf.keras.Sequential([
    tf.keras.layers.Dense(25, activation='relu'),
    tf.keras.layers.Dense(25, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
  ])


# In[30]:


model.compile(optimizer="adam", 
              loss ="binary_crossentropy", 
              metrics=["accuracy"])


# In[31]:


model.fit(X_train, y_train, 
         epochs=30, 
         batch_size=20,
         validation_data=(X_test, y_test)
         )


# In[32]:


# `rankdir='LR'` is to make the graph horizontal.
tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")


# In[33]:


loss, accuracy = model.evaluate(X_test, y_test)

print("Accuracy", accuracy)


# In[34]:


model.save('classifier_mobile_hd')


# In[35]:


reloaded_model = tf.keras.models.load_model('classifier_mobile_hd')


# In[36]:


predictions = reloaded_model.predict(X_train)


# # SHAP

# In[37]:


explainer = shap.KernelExplainer(model, X_train.iloc[:50,:])


# In[38]:


shap_values = explainer.shap_values(X_train.iloc[20,:], nsamples=500)


# In[39]:


shap.force_plot(explainer.expected_value, shap_values[0], X_train.iloc[25,:])


# In[40]:


shap_values50 = explainer.shap_values(X_train.iloc[50:100,:], nsamples=500)


# In[41]:


shap.force_plot(explainer.expected_value, shap_values50[0], X_train.iloc[50:100,:])


# Shapley Additive exPlanations oder SHAP ist ein Ansatz, der in der Spieltheorie verwendet wird. Mit SHAP wird die Ausgabe seines maschinellen Lernmodells erklärt.
# 
# In dem obigen Modell werden Merkmale aufgezeigt, die dazu beitragen, die Modellausgabe zu steigern. Merkmale bzw. Vorhersagen die sich positiv auswirken werde in Rot dargestellt und Vorhersagen, die sich negativ auswirken werden in Blau dargestellt.
