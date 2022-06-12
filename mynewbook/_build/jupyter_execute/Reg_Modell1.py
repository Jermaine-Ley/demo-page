#!/usr/bin/env python
# coding: utf-8

# # Regressionsmodell

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import neighbors, datasets, svm, tree
import warnings 
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv("car_prices.csv",nrows=400000)
print(df) # Ausgabe 


# In[3]:


#Die Funktion head() wird verwendet, um die ersten n Zeilen zu erhalten.
#Diese Funktion gibt die ersten n Zeilen des Objekts auf der Grundlage der Position zurück. Sie ist nützlich, um schnell zu testen, ob das Objekt die richtige Art von Daten enthält.
df.head()


# In[4]:


#Das shape-Attribut von pandas.DataFrame speichert die Anzahl der Zeilen und Spalten als Tupel 
df.shape 


# In[5]:


#Die Methode info() von pandas.DataFrame kann Informationen wie die Anzahl der Zeilen und Spalten, 
#den gesamten Speicherverbrauch, den Datentyp jeder Spalte und die Anzahl der Nicht-NaN-Elemente anzeigen.

df.info()


# In[6]:


#Fehlende Werte erkennen. Gibt ein boolesches Objekt zurück, das angibt, ob die Werte NA sind 
df.isna().sum().sort_values(ascending=False)


# In[7]:


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


# In[8]:


categorical_columns


# In[9]:


continous_columns


# In[10]:


discrete_columns


# In[11]:


#Füllen von fehlenden Werten kontinuierlicher Spalten mit dem Median
for x in continous_columns:
  df[x].fillna(df[x].median(),inplace=True)


# In[12]:


#Füllen fehlender Werte kategorischer Spalten mit Modus
for x in categorical_columns:
  df[x].fillna(df[x].mode()[0],inplace=True)


# In[13]:


df.isna().sum().sort_values(ascending=False)


# In[14]:


#Wir haben keine Missing Values mehr, somit ist die Datei sauber und wir können beginnen.


# In[15]:


#Mit einem Boxplot werden die Ausreißer überprüft.
df[continous_columns].plot(kind='box',subplots=True,layout=(2,3),figsize=(14,8))
plt.show()


# In[16]:


# Funktion zur Rückgabe des Index für die Spalte, deren Datenpunkte größer als die angegebene Grenze sind
def outs(col,limit):
    index = []
    index = df[df[col]>limit].index

    return index


# In[17]:


# Dieser Index hat Datenpunkte, die sehr weit von der Gruppe der Datenpunkte entfernt sind.
# Ersetzt werden diese Punkte durch Werte in der Nähe des Clusters
ind = outs('odometer',900000)
ind


# In[18]:


value = round(np.percentile(df.odometer,99),1)
value


# In[19]:


df.loc[ind,'mmr'] = value


# In[20]:


# Dieser Index hat Datenpunkte, die sehr weit von der Gruppe der Datenpunkte entfernt sind.
# Es wurden Punkte ersetzt durch Werte die in der Nähe des Clusters sind.
ind = outs('sellingprice',100000)
ind


# In[21]:


value = round(np.percentile(df.sellingprice,99),1)
value


# In[22]:


df.loc[ind,'sellingprice'] = value


# In[23]:


# Die Außreiser sind damit abgeschlossen
# Label-Kodierung


# In[24]:


categorical_columns


# In[25]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for x in categorical_columns:
  df[x] = le.fit_transform(df[x])


# In[26]:


#Feature Selection
X = df.copy()


# In[27]:


X.drop('sellingprice',inplace=True,axis=1)


# In[28]:


Y = df.sellingprice


# In[29]:


from sklearn.ensemble import ExtraTreesRegressor
et = ExtraTreesRegressor()
et.fit(X,Y)


# In[30]:


zip(et.feature_importances_,X.columns)


# In[31]:


imp_col = pd.DataFrame(zip(et.feature_importances_,X.columns),columns=['Importance','Columns'])


# In[32]:


imp_col.sort_values(by='Importance',ascending=False).head()


# In[33]:


#Auswahl der ersten 5 Spalten
X = X[['mmr','year','condition','odometer','vin']]


# In[34]:


X


# In[35]:


#Modelbau und Splittung un Trainings-und Testdaten
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, 
random_state=10)


# In[36]:


#Lineare Regression
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train, Y_train)
Y_pred=lr.predict(X_test)


# In[37]:


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


# In[38]:


#Streudiagramm der tatsächlichen Werte gegenüber den vorhergesagten Werten
sns.scatterplot(x=Y_test,y=Y_pred)


# In[39]:


sns.distplot(Y_test-Y_pred)


# In[ ]:


# #print first five rows of dataset
# print(df.head())

# #print shape of dataframe
# print("Shape of data frame: ",df.shape)

# #print descriptive statistics of this dataset
# print("\n",df.describe())

# #Show unique values for "method" using "unique" method
# print("\nUnique values for method: ",df['state'].unique())


# In[ ]:


# #Calculate correlation between all attributs
# corr = df[['year', 'make', 'model', 'trim', 'body', 'transmission', 'vin', 'state', 'condition', 'odometer', 'color', 'interior', 'seller', 'mmr', 'sellingprice', 'saledate']].corr()

# #generate a custom diverging colormap
# cmap = sns.diverging_palette(230, 20, as_cmap=True)

# #draw Heatmap with correlations
# sns.heatmap(corr,cmap=cmap)
# plt.show()

