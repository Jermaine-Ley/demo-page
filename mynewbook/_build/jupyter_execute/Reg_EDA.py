#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis (EDA) Auktion

# In[38]:


# Importing Libraries

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt             #visualisation
import seaborn as sns                       #visualisation

get_ipython().run_line_magic('matplotlib', 'inline')


# In[39]:


# Loading the Data into the Data Frame
df = pd.read_csv('car_prices.csv',error_bad_lines=False,warn_bad_lines=True)

# So werden die obersten 5 Zeilen angezeigt
df.head(5)


# In[40]:


# Um die untersten 5 Zeilen anzuzeigen
df.tail(3)


# In[41]:


# So prüfen man die Datentypen
df.dtypes


# Hier werde ich einige Zeilen löschen, die für die Daten unerheblich sind.
# 

# In[42]:


# So lässt man die irrelevanten Spalten aus.
df = df.drop(['vin'], axis=1)
df.head(5)


# 
# Manchmal können Daten doppelte Zeilen enthalten. Hier habe ich die Anzahl der doppelten Zeilen überprüft, aber es gab keine doppelten Zeilen.
# 

# In[43]:


# To find the number of duplicate rows

duplicate_rows_df = df[df.duplicated()]
print("number of duplicate rows: ", duplicate_rows_df.shape)


# Diese Daten enthalten mehrere Nullwerte. Ich zeige die Nullwerte im Dataset an.
# 

# In[44]:


# So finden Sie die Anzahl der doppelten Zeilen
print(df.isnull().sum())     # Ermittlung der Anzahl der Nullwerte


# Hier lasse ich die Nullwerte weg und lösche sie. Da es sich um einen sehr großen Datensatz handelt, wirkt sich das Löschen einiger Werte nicht auf die gesamten Daten aus.

# In[45]:


df = df.dropna()    # Die fehlenden Werte werden gestrichen.
df.count()


# In[46]:


print(df.isnull().sum())   # Nach dem löschen der Werte


# Auffinden der Ausreißer in den Daten. Hier habe ich die Ausreißer in Kilometerzähler und Preis aufgezeichnet. Da die Ausreißer bei Kilometerzähler und Preis einen gewissen Einfluss auf die Daten haben, lösche ich die Ausreißer nicht. Denn der Preis eines Fahrzeugs und die gefahrene Strecke hängen von vielen anderen Faktoren ab.

# In[47]:


# Dies sind die Schritte für den Box Plot

sns.boxplot(x=df['odometer'])


# In[57]:


bgggggggggggggggggggggggggggggggggggdddddddddddddddddddddddffff


# In[49]:


# Dies sind die Schritte für den Box Plot

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# Es wurde ein Diagramm mit der Anzahl der Fahrzeuge im Vergleich zu den Marken erstellt, um die wichtigsten Marken in der Auktion zu ermitteln. Dabei hatte Ford die meisten Autos.
# 

# In[50]:


# Dies sind die Schritte für das Balkendiagramm

df.make.value_counts().nlargest(40).plot(kind='bar', figsize=(10,5))
plt.title("Anzahl von Autos in Bezug auf die Marke(make)")
plt.ylabel('Anzahl der Autos')
plt.xlabel('Marken');


# 
# # Insights finden
# 
# Hier habe ich ein Diagramm erstellt, das die Anzahl der verkauften Fahrzeuge in Abhängigkeit von ihrem Preis zeigt. Die meisten Fahrzeuge wurden in einem Bereich von 10 000 bis 20 000 verkauft.
# 

# In[51]:


# These are for plotting Histogram

plt.figure(figsize=(15,8))
plt.hist(df['sellingprice'],bins=200)
plt.title('Verteilung des Verkaufspreises',fontsize=20)
plt.xticks(np.arange(0,df['sellingprice'].max(),10000))
plt.xlabel('Verkaufspreis',fontsize=12)
plt.ylabel('Freq',fontsize=12)


# In[52]:


df['sellingprice'].describe()   # Beschreibung des obigen Diagramms


# Erstellung eines Diagramms, das zeigt, wie der Kilometerstand den Verkaufspreis von Gebrauchtwagen beeinflusst. Fahrzeuge mit einem Kilometerstand von weniger als 20 Kilometern erzielen den höchsten Preis beim Verkauf. Mit zunehmender Fahrleistung sinkt der Verkaufspreis.
# 

# In[53]:


# Diese sind für das Plotten von Streudiagrammen gedacht.

plt.figure(figsize=(20,8))
g=sns.scatterplot(x='odometer',y='sellingprice',data=df)
g.set_title('Kilometerstand vS Korrelation der Verkaufspreise',fontsize=20)
g.set_xlabel('Killometerstand',fontsize=10)
g.set_ylabel('Verkaufspreis',fontsize=10)
xlabels=['{:,.2f}'.format(x)+'k' for x in g.get_xticks()/10e3]
ylabels=['{:,.2f}'.format(y)+'k' for y in g.get_yticks()/10e3]
g.set_xticklabels(xlabels);
g.set_yticklabels(ylabels);


# In[54]:


# Dies sind die Schritte für das Balkendiagramm

plt.figure(figsize=(19,6))

df['year'].value_counts().plot(kind='bar')

plt.title('Anzahl der verkauften Autos nach Jahr',fontsize=20)
plt.xlabel('Jahr',fontsize=15)
plt.ylabel('Gesamte Autos',fontsize=15)


# In[55]:


# These are for plotting Bar Plot

plt.figure(figsize=(20,7))
sns.barplot(x='year',y='sellingprice',data=df)
plt.title('Modelle nach Verkaufspreis')
plt.xlabel('Jahr')
plt.ylabel('Verkaufspreis')


# # Schlussfolgerung
# 
# In diesem Notizbuch habe ich EDA auf dem Datensatz für Autoauktionen durchgeführt, um die Faktoren zu verstehen, die die Versteigerung von Gebrauchtwagen beeinflussen. Es handelte sich um einen großen Datensatz mit etwa 558T Zeilen. Ich habe die Nullwerte und irrelevanten Spalten gelöscht, um ein besseres Verständnis der Daten zu erhalten.
# 
# In dieser Analyse habe ich einige Diagramme erstellt, um die Verteilung des Verkaufspreises nach verschiedenen Faktoren wie Kilometerstand, Baujahr des Modells und Marken des Fahrzeugs zu zeigen.
# 
# Aus dieser Analyse konnte ich entnehmen, dass Faktoren wie Kilometerstand (die vom Fahrzeug zurückgelegte Strecke), Fahrzeugmarken und Baujahr des Modells die Versteigerung von Gebrauchtwagen hauptsächlich beeinflussen. Autos mit geringem Kilometerstand erzielen bei der Versteigerung einen guten Preis.
# 
