#!/usr/bin/env python
# coding: utf-8

# # Keras Tuner

# In[ ]:


import keras_tuner as kt

from tensorflow import keras
from tensorflow.keras import layers

from tensorboard import notebook
from tensorboard.plugins.hparams import api as hp

import numpy as np
import datetime

get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[ ]:


def build_model(hp):

    model = keras.Sequential()
    
    model.add(layers.Flatten())
    
    model.add(
        layers.Dense(
            # Define the hyperparameter.
            units = hp.Int("units", min_value=32, 
                                    max_value=512, 
                                    step=32),
            activation = "relu",
        )
    )
    model.add(layers.Dense(10, activation="softmax"))
    
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"],
    
    )
    return model


# In[ ]:


build_model(kt.HyperParameters())


# In[ ]:


def build_model(hp):
    
    model = keras.Sequential()

    model.add(layers.Flatten())
    
    model.add(
        layers.Dense(
            # Tune number of units.
            units=hp.Int("units", min_value=32, max_value=512, step=32),
            # Tune the activation function to use.
            activation=hp.Choice("activation", ["relu", "tanh"]),
        )
    )
    
    # Tune whether to use dropout.
    if hp.Boolean("dropout"):
        model.add(layers.Dropout(rate=0.25))
    model.add(layers.Dense(10, activation="softmax"))
    
    # Define the optimizer learning rate as a hyperparameter.
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    
    return model


# In[ ]:


build_model(kt.HyperParameters())


# In[ ]:


hp = kt.HyperParameters()

print(hp.Int("units", min_value=32, max_value=512, step=32))


# # Variable lists¶

# In[ ]:


# list of all numerical data
list_num = df.select_dtypes(include=[np.number]).columns.tolist()

# list of all categorical data
list_cat = df.select_dtypes(include=['category']).columns.tolist()


# In[ ]:


# define outcome variable as y_label
y_label = 'ram'

# select features
features = df.drop(columns=[y_label]).columns.tolist()

# create feature data for data splitting
X = df[features]

# list of numeric features
feat_num = X.select_dtypes(include=[np.number]).columns.tolist()

# list of categorical features
feat_cat = X.select_dtypes(include=['category']).columns.tolist() 

# create response for data splitting
y = df[y_label]


# # Start search

# In[ ]:


tuner = kt.RandomSearch(
    hypermodel=build_model,
    objective="val_accuracy",
    max_trials=3,
    executions_per_trial=2,
    overwrite=True,
    directory="tmp",
    project_name="Mobile_Price",
)


# In[ ]:


tuner.search_space_summary()


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


(x, y), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x[:-10000]
x_val = x[-10000:]
y_train = y[:-10000]
y_val = y[-10000:]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




