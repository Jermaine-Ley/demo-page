#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis (EDA) Auktion

# In[4]:


# Importing Libraries

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt             #visualisation
import seaborn as sns                       #visualisation

get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


# Loading the Data into the Data Frame
df = pd.read_csv('car_prices.csv',error_bad_lines=False,warn_bad_lines=True)

# To display the top 5 rows
df.head(5)


# In[6]:


# To display the botton 5 rows
df.tail(3)


# In[7]:


# To check the types of data
df.dtypes



# Hier werde ich einige Zeilen löschen, die für die Daten unerheblich sind.
# 

# In[8]:


# To drop the irrelevant columns

df = df.drop(['vin', 'mmr'], axis=1)
df.head(5)


# 
# Manchmal können Daten doppelte Zeilen enthalten. Hier habe ich die Anzahl der doppelten Zeilen überprüft, aber es gab keine doppelten Zeilen.
# 

# In[9]:


# To find the number of duplicate rows

duplicate_rows_df = df[df.duplicated()]
print("number of duplicate rows: ", duplicate_rows_df.shape)



# Diese Daten enthalten mehrere Nullwerte. Ich zeige die Nullwerte im Dataset an.
# 

# In[10]:


#  To Drop the missing or null values
print(df.isnull().sum())     # Finding the number of Null values


# Hier lasse ich die Nullwerte weg und lösche sie. Da es sich um einen sehr großen Datensatz handelt, wirkt sich das Löschen einiger Werte nicht auf die gesamten Daten aus.

# In[11]:


df = df.dropna()    # Dropping the missing values.
df.count()


# In[12]:


print(df.isnull().sum())   # After dropping the values



# Auffinden der Ausreißer in den Daten. Hier habe ich die Ausreißer in Kilometerzähler und Preis aufgezeichnet. Da die Ausreißer bei Kilometerzähler und Preis einen gewissen Einfluss auf die Daten haben, lösche ich die Ausreißer nicht. Denn der Preis eines Fahrzeugs und die gefahrene Strecke hängen von vielen anderen Faktoren ab.

# In[14]:


# These are the steps for Box Plot

sns.boxplot(x=df['odometer'])


# In[17]:


# These are the steps for Box Plot

sns.boxplot(x=df['sellingprice'])


# In[19]:


# Nothing to worry, If you don't understand this.
# These are the steps for Box Plot

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# Es wurde ein Diagramm mit der Anzahl der Fahrzeuge im Vergleich zu den Marken erstellt, um die wichtigsten Marken in der Auktion zu ermitteln. Und Ford war am meisten.
# 

# In[25]:


# These are the steps for Bar Diagram

df.make.value_counts().nlargest(40).plot(kind='bar', figsize=(10,5))
plt.title("Number of cars by Brands (make)")
plt.ylabel('Number of cars')
plt.xlabel('Brands');


# 
# Insights finden
# 
# Hier habe ich ein Diagramm erstellt, das die Anzahl der verkauften Fahrzeuge in Abhängigkeit von ihrem Preis zeigt. Die meisten Fahrzeuge wurden in einem Bereich von 10 000 bis 20 000 verkauft.
# 

# In[28]:


# Nothing to worry, If you don't understand this.
# These are for plotting Histogram
# Comment if you have any doubts

plt.figure(figsize=(15,8))
plt.hist(df['sellingprice'],bins=200)
plt.title('Distribution of Selling Price',fontsize=20)
plt.xticks(np.arange(0,df['sellingprice'].max(),10000))
plt.xlabel('Selling Price',fontsize=12)
plt.ylabel('Freq',fontsize=12)


# In[30]:


df['sellingprice'].describe()   # Describing the above graph



# Erstellung eines Diagramms, das zeigt, wie der Kilometerstand den Verkaufspreis von Gebrauchtwagen beeinflusst. Fahrzeuge mit einem Kilometerstand von weniger als 20 Kilometern erzielen den höchsten Preis beim Verkauf. Mit zunehmender Fahrleistung sinkt der Verkaufspreis.
# 

# In[32]:


# Nothing to worry, If you don't understand this.
# These are for plotting Scatter Plot
# Comment if you have any doubts

plt.figure(figsize=(20,8))
g=sns.scatterplot(x='odometer',y='sellingprice',data=df)
g.set_title('Odometer vS Selling Price Correlation',fontsize=20)
g.set_xlabel('Odometer',fontsize=10)
g.set_ylabel('Selling Price',fontsize=10)
xlabels=['{:,.2f}'.format(x)+'k' for x in g.get_xticks()/10e3]
ylabels=['{:,.2f}'.format(y)+'k' for y in g.get_yticks()/10e3]
g.set_xticklabels(xlabels);
g.set_yticklabels(ylabels);


# In[33]:


# These are the steps for Bar Diagram

plt.figure(figsize=(19,6))

df['year'].value_counts().plot(kind='bar')

plt.title('Number of Cars Sold by year of Brands',fontsize=20)
plt.xlabel('Year',fontsize=15)
plt.ylabel('Total Cars',fontsize=15)


# In[34]:


# Nothing to worry, If you don't understand this.
# These are for plotting Bar Plot
# Comment if you have any doubts

plt.figure(figsize=(20,7))
sns.barplot(x='year',y='sellingprice',data=df)
plt.title('Models by Selling Price')
plt.xlabel('Year')
plt.ylabel('Selling Price')


# # Schlussfolgerung
# 
# In diesem Notizbuch habe ich EDA auf dem Datensatz für Autoauktionen durchgeführt, um die Faktoren zu verstehen, die die Versteigerung von Gebrauchtwagen beeinflussen. Es handelte sich um einen großen Datensatz mit etwa 558T Zeilen. Ich habe die Nullwerte und irrelevanten Spalten gelöscht, um ein besseres Verständnis der Daten zu erhalten.
# 
# In dieser Analyse habe ich einige Diagramme erstellt, um die Verteilung des Verkaufspreises nach verschiedenen Faktoren wie Kilometerstand, Baujahr des Modells und Marken des Fahrzeugs zu zeigen.
# 
# Aus dieser Analyse konnte ich entnehmen, dass Faktoren wie Kilometerstand (die vom Fahrzeug zurückgelegte Strecke), Fahrzeugmarken und Baujahr des Modells die Versteigerung von Gebrauchtwagen hauptsächlich beeinflussen. Autos mit geringem Kilometerstand erzielen bei der Versteigerung einen guten Preis.
# 
