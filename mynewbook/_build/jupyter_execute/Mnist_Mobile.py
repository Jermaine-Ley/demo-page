#!/usr/bin/env python
# coding: utf-8

# # MNIST

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[3]:


import keras_tuner as kt

from tensorflow import keras
from tensorflow.keras import layers

from tensorboard import notebook
from tensorboard.plugins.hparams import api as hp

import numpy as np
import datetime

get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[7]:


mnist_train = pd.read_csv("Mobile_Price_train.csv")
mnist_test = pd.read_csv("Mobile_Price_test.csv")


# In[8]:


mnist_train


# In[11]:


y_train = mnist_train["id"].copy().to_numpy()
X_train = mnist_train.drop(columns=["id"]).to_numpy()

print("The training digits data:\n", X_train)
print("Digit labels: ", y_train)

# Similarly for the test set
y_test = mnist_test["id"].copy().to_numpy()
X_test = mnist_test.drop(columns=["id"]).to_numpy()

