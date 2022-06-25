#!/usr/bin/env python
# coding: utf-8

# # KerasTuner

# *The following code is based on ["Getting started with KerasTuner
# "](https://keras.io/guides/keras_tuner/getting_started/) from Luca Invernizzi, James Long, Francois Chollet, Tom O'Malley and Haifeng Jin.*

# 
# [KerasTuner](https://keras.io/guides/keras_tuner/getting_started/) is an easy-to-use, scalable hyperparameter optimization framework that solves the pain points of hyperparameter search. 
# 
# - It is a general-purpose hyperparameter tuning library.
# - It has strong integration with Keras workflows, but it isn't limited to them
# - You can use it to tune scikit-learn models, or anything else. 
# 
# KerasTuner comes with: 
# 
# - [Bayesian Optimization](https://keras.io/api/keras_tuner/tuners/bayesian/) (see this [blog](https://distill.pub/2020/bayesian-optimization/) for detailed explanations), 
# - [Hyperband](https://keras.io/api/keras_tuner/tuners/hyperband/) (see this [paper](https://jmlr.org/papers/v18/16-558.html))
# - [Random Search algorithms](https://keras.io/api/keras_tuner/tuners/random/) 

# ## Installation

# - KerasTuner requires Python 3.6+ and TensorFlow 2.0+.

# - Installation with pip: 
# 
# ```bash
# pip install keras-tuner --upgrade
# ```
# 
# - Installation with conda
# 
# ```bash
# conda install -c conda-forge keras-tuner
# ```

# ## Setup

# In[23]:


import keras_tuner as kt

from tensorflow import keras
from tensorflow.keras import layers

from tensorboard import notebook
from tensorboard.plugins.hparams import api as hp

import numpy as np
import datetime

get_ipython().run_line_magic('load_ext', 'tensorboard')


# ## Define search space

# - Write a function that creates and returns a Keras model. 
# - Use the `hp` argument to define the hyperparameters during model creation.

# In[24]:


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


# You can quickly test if the model builds successfully.
# 
# 

# In[25]:


build_model(kt.HyperParameters())


# - There are many other types of hyperparameters as well. 
# - We can define multiple hyperparameters in the function. 
# - In the following code, we tune the whether to 
#   - use a Dropout layer with hp.Boolean(), 
#   - tune which activation function to use with hp.Choice(), 
#   - tune the learning rate of the optimizer with hp.Float().

# In[26]:


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


# In[27]:


build_model(kt.HyperParameters())


# - As shown below, the hyperparameters are actual values. 
# - In fact, they are just functions returning actual values. 
# - For example, `hp.Int()` returns an int value. 
# - Therefore, you can put them into variables, for loops, or if conditions.

# In[28]:


hp = kt.HyperParameters()

print(hp.Int("units", min_value=32, max_value=512, step=32))


# ## Start search

# - After defining the search space, we need to select a tuner class to run the search. 
# - You may choose from `RandomSearch`, `BayesianOptimization` and `Hyperband`, which correspond to different tuning algorithms. 
# - Here we use `RandomSearch` as an example.
# 

# To initialize the tuner, we need to specify several arguments in the initializer:
# 
# - `hypermodel`. The model-building function, which is build_model in our case.
# - `objective`. The name of the objective to optimize (whether to minimize or maximize is automatically inferred for built-in metrics). 
# - `max_trials`. The total number of trials to run during the search.
# - `executions_per_trial`. The number of models that should be built and fit for each trial. Different trials have different hyperparameter values. The executions within the same trial have the same hyperparameter values. The purpose of having multiple executions per trial is to reduce results variance and therefore be able to more accurately assess the performance of a model. If you want to get results faster, you could set executions_per_trial=1 (single round of training for each model configuration).
# - `overwrite`. Control whether to overwrite the previous results in the same directory or resume the previous search instead. Here we set overwrite=True to start a new search and ignore any previous results.
# - `directory`. A path to a directory for storing the search results.
# -`project_name`. The name of the sub-directory in the directory.

# In[29]:


tuner = kt.RandomSearch(
    hypermodel=build_model,
    objective="val_accuracy",
    max_trials=3,
    executions_per_trial=2,
    overwrite=True,
    directory="tmp",
    project_name="hello_world",
)


# You can print a summary of the search space:

# In[30]:


tuner.search_space_summary()


# ## Data

# - Before starting the search, let's prepare the data.
# - We use the MNIST dataset.
# - Since we use hyperparameter tuning, we use training, evaluation and test data.

# In[31]:


(x, y), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x[:-10000]
x_val = x[-10000:]
y_train = y[:-10000]
y_val = y[-10000:]


# In[32]:


x_train


# In[33]:


x_train = np.expand_dims(x_train, -1).astype("float32") / 255.0


# In[34]:


x_val = np.expand_dims(x_val, -1).astype("float32") / 255.0
x_test = np.expand_dims(x_test, -1).astype("float32") / 255.0

num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# - Then, start the search for the best hyperparameter configuration. 
# - All the arguments passed to search is passed to model.fit() in each execution. 
# - Remember to pass validation_data to evaluate the model.

# - During the search, the model-building function is called with different hyperparameter values in different trial. 
# - In each trial, the tuner would generate a new set of hyperparameter values to build the model. 
# - The model is then fit and evaluated. 
# - The metrics are recorded. 
# - The tuner progressively explores the space and finally finds a good set of hyperparameter values.

# - To use TensorBoard, we need to pass a keras.callbacks.TensorBoard instance to the callbacks.

# In[35]:


# Create TensorBoard folders
log_dir = "tmp/tb_logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tuner.search(
    x_train,
    y_train,
    epochs=2,
    validation_data=(x_val, y_val),
    # Use the TensorBoard callback.
    # The logs will be write to "/tmp/tb_logs".
    callbacks=[keras.callbacks.TensorBoard(log_dir=log_dir)],
)


# ## Query results

# - When search is over, you can retrieve the best model(s). 
# - The model is saved at its best performing epoch evaluated on the validation_data.

# In[36]:


get_ipython().run_line_magic('tensorboard', '--logdir /tmp/tb_logs')


# If TensorBoard does not open a window, try to use the following code to start TensorBoard:

# In[37]:


# View open TensorBoard instances
notebook.list() 


# In[38]:


# Start TensorBoard by providing the right port
notebook.display(port=6006, height=1000)


# The following content is based on ["Visualize the hyperparameter tuning process"](https://keras.io/guides/keras_tuner/visualize_tuning/) by Jin Haifeng:
# 
# - You have access to all the common features of the TensorBoard. 
# - For example, you can view the loss and metrics curves and visualize the computational graph of the models in different trials.
# - In addition to these features, we also have a `HParams` tab, in which there are three views:
# 
# 1. In the table view, you can view the different trials in a table with the different hyperparameter values and evaluation metrics. On the left side, you can specify the filters for certain hyperparameters. 
#   
# 1. It also provides parallel coordinates view and scatter plot matrix view. They are just different visualization methods for the same data:
#   - In the parallel coordinates view, each colored line is a trial. The axes are the hyperparameters and evaluation metrics.
#   - In the scatter plot matrix view, each dot is a trial. The plots are projections of the trials on planes with different hyperparameter and metrics as the axes.

# In[39]:


# Get the top 2 models.
models = tuner.get_best_models(num_models=2)
# Get best model
best_model = models[0]


# In[40]:


# Build the model.
# Needed for `Sequential` without specified `input_shape`.
best_model.build(input_shape=(None, 28, 28))

best_model.summary()


# You can also print a summary of the search results.

# In[41]:


tuner.results_summary()


# - You will find detailed logs, checkpoints, etc, in your specified folder.
# 
# - You can also visualize the tuning results using TensorBoard and HParams plugin. For more information, please following this link.
