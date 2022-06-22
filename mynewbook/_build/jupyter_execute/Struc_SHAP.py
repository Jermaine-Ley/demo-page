#!/usr/bin/env python
# coding: utf-8

# # SHAP mit strukturierten Daten (Classification)

# In[111]:


import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import shap

tf.__version__


# In[112]:


# print the JS visualization code to the notebook
shap.initjs()


# In[113]:


df = pd.read_csv('Mobile_Price_train.csv',error_bad_lines=False,warn_bad_lines=True)
print(df) # Ausgabe 


# In[114]:


df.info()


# In[ ]:





# In[115]:


# make target variable
y = df.pop('price_range')


# In[116]:


# prepare features
list_numerical = ['ram', 'battery_power', 'touch_screen']

X = df[list_numerical]


# In[117]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[118]:


scaler = StandardScaler().fit(X_train[list_numerical]) 

X_train[list_numerical] = scaler.transform(X_train[list_numerical])
X_test[list_numerical] = scaler.transform(X_test[list_numerical])


# In[119]:


model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
  ])


# In[120]:


model.compile(optimizer="adam", 
              loss ="binary_crossentropy", 
              metrics=["accuracy"])


# In[121]:


model.fit(X_train, y_train, 
         epochs=15, 
         batch_size=13,
         validation_data=(X_test, y_test)
         )


# In[122]:


# `rankdir='LR'` is to make the graph horizontal.
tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")


# In[123]:


loss, accuracy = model.evaluate(X_test, y_test)

print("Accuracy", accuracy)


# In[124]:


model.save('classifier_mobile_hd')


# In[125]:


reloaded_model = tf.keras.models.load_model('classifier_mobile_hd')


# In[126]:


predictions = reloaded_model.predict(X_train)


# In[127]:


print(
    "This particular patient had a %.1f percent probability "
    "of having a heart disease, as evaluated by our model." % (100 * predictions[0][0],)
)


# In[128]:


explainer = shap.KernelExplainer(model, X_train.iloc[:50,:])


# In[129]:


shap_values = explainer.shap_values(X_train.iloc[20,:], nsamples=500)


# In[130]:


shap.force_plot(explainer.expected_value, shap_values[0], X_train.iloc[20,:])


# In[131]:


shap_values50 = explainer.shap_values(X_train.iloc[50:100,:], nsamples=500)


# In[132]:


shap.force_plot(explainer.expected_value, shap_values50[0], X_train.iloc[50:100,:])


# In[ ]:





# 
