#!/usr/bin/env python
# coding: utf-8

# # Regressionsmodell

# In[3]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import neighbors, datasets, svm, tree
import warnings 
warnings.filterwarnings('ignore')


# In[4]:


df = pd.read_csv("car_prices.csv",nrows=400000)
print(df) # Ausgabe 


# In[5]:


#Die Funktion head() wird verwendet, um die ersten n Zeilen zu erhalten.
#Diese Funktion gibt die ersten n Zeilen des Objekts auf der Grundlage der Position zurück. Sie ist nützlich, um schnell zu testen, ob das Objekt die richtige Art von Daten enthält.
df.head()


# In[6]:


#Das shape-Attribut von pandas.DataFrame speichert die Anzahl der Zeilen und Spalten als Tupel 
df.shape 


# In[7]:


#Die Methode info() von pandas.DataFrame kann Informationen wie die Anzahl der Zeilen und Spalten, 
#den gesamten Speicherverbrauch, den Datentyp jeder Spalte und die Anzahl der Nicht-NaN-Elemente anzeigen.

df.info()


# In[8]:


#Fehlende Werte erkennen. Gibt ein boolesches Objekt zurück, das angibt, ob die Werte NA sind 
df.isna().sum().sort_values(ascending=False)


# In[9]:


#Ab diesen Abschnitt werden die fehlenden Werte je nach Datentyp der Spalte. 
# \ Kategorische Spalten -> Modus \ Kontinuierliche Spalten -> Mittelwert/Median \ Diskrete Spalten -> Modus


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


# In[10]:


categorical_columns


# In[11]:


continous_columns


# In[12]:


discrete_columns


# In[13]:


#Füllen von fehlenden Werten kontinuierlicher Spalten mit dem Median
for x in continous_columns:
  df[x].fillna(df[x].median(),inplace=True)


# In[14]:


#Füllen fehlender Werte kategorischer Spalten mit Modus
for x in categorical_columns:
  df[x].fillna(df[x].mode()[0],inplace=True)


# In[15]:


df.isna().sum().sort_values(ascending=False)


# In[16]:


#Wir haben keine Missing Values mehr, somit ist die Datei sauber und wir können beginnen.


# In[17]:


#Mit einem Boxplot werden die Ausreißer überprüft.
df[continous_columns].plot(kind='box',subplots=True,layout=(2,3),figsize=(14,8))
plt.show()


# In[18]:


# Funktion zur Rückgabe des Index für die Spalte, deren Datenpunkte größer als die angegebene Grenze sind
def outs(col,limit):
    index = []
    index = df[df[col]>limit].index

    return index


# In[19]:


# Dieser Index hat Datenpunkte, die sehr weit von der Gruppe der Datenpunkte entfernt sind.
# Ersetzt werden diese Punkte durch Werte in der Nähe des Clusters
ind = outs('odometer',900000)
ind


# In[20]:


value = round(np.percentile(df.odometer,99),1)
value


# In[21]:


df.loc[ind,'mmr'] = value


# In[22]:


# Dieser Index hat Datenpunkte, die sehr weit von der Gruppe der Datenpunkte entfernt sind.
# Es wurden Punkte ersetzt durch Werte die in der Nähe des Clusters sind.
ind = outs('sellingprice',100000)
ind


# In[23]:


value = round(np.percentile(df.sellingprice,99),1)
value


# In[24]:


df.loc[ind,'sellingprice'] = value


# In[25]:


# Die Außreiser sind damit abgeschlossen
# Label-Kodierung


# In[26]:


categorical_columns


# In[27]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for x in categorical_columns:
  df[x] = le.fit_transform(df[x])


# In[28]:


#Feature Selection
X = df.copy()


# In[29]:


X.drop('sellingprice',inplace=True,axis=1)


# In[30]:


Y = df.sellingprice


# In[32]:


from sklearn.ensemble import ExtraTreesRegressor
et = ExtraTreesRegressor()
et.fit(X,Y)


# In[33]:


zip(et.feature_importances_,X.columns)


# In[34]:


imp_col = pd.DataFrame(zip(et.feature_importances_,X.columns),columns=['Importance','Columns'])


# In[35]:


imp_col.sort_values(by='Importance',ascending=False).head()


# In[36]:


#Auswahl der ersten 5 Spalten
X = X[['mmr','year','condition','odometer','vin']]


# In[37]:


X


# In[38]:


#Modelbau und Splittung un Trainings-und Testdaten
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, 
random_state=10)


# In[39]:


#Lineare Regression
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train, Y_train)
Y_pred=lr.predict(X_test)


# In[40]:


# Evaluation
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
import numpy as np

lracc = r2_score(Y_test,Y_pred)
print('Score ->',lracc)

lrmae = mean_absolute_error(Y_test,Y_pred)
print('MAE ->',lrmae)

lrmse = mean_squared_error(Y_test,Y_pred)
print('MSE ->',lrmse)

lrrmse = np.sqrt(mean_squared_error(Y_test,Y_pred))
print('RMSE ->',lrrmse)

adjusted_r_squared = 1 - (1-lracc)*(len(Y)-1)/(len(Y)-X.shape[1]-1) 
print('Adjusted R2 ->',adjusted_r_squared)


# In[41]:


#Streudiagramm der tatsächlichen Werte gegenüber den vorhergesagten Werten
sns.scatterplot(x=Y_test,y=Y_pred)


# In[42]:


sns.distplot(Y_test-Y_pred)

