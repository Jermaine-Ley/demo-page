#!/usr/bin/env python
# coding: utf-8

# # Was beeinflusst den Verkaufspreis?

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import re

from collections import Counter
import os


# In[2]:


df = pd.read_csv('car_prices.csv',error_bad_lines=False,warn_bad_lines=True)
print(df) # Ausgabe 


# In[3]:


df.shape


# Die Gesamtzahl der Zeilen beträgt 558 Tausend mit 16 Spalten.

# In[4]:


#Die Funktion head() wird verwendet, um die ersten n Zeilen zu erhalten.
#Diese Funktion gibt die ersten n Zeilen des Objekts auf der Grundlage der Position zurück. Sie ist nützlich, um schnell zu testen, ob das Objekt die richtige Art von Daten enthält.
df.head()


# In[5]:


#Gibt ein Tupel zurück, das die Dimensionalität des DataFrame angibt.
(df.isnull().sum()/df.shape[0])*100


# Betrachtet man die Nullwerte, so sind mit Ausnahme der Übertragung (11 %) alle anderen vernachlässigbar. 
# Der Einfachheit halber lassen wir diese Nullwerte weg und führen unsere Analyse durch. Prüfen wir auf Duplikate,

# In[6]:


df.duplicated().sum()


# No rows are duplicated.
# 

# In[7]:


#Gibt die Datentypen im DataFrame zurück.

# Dies gibt eine Serie mit dem Datentyp jeder Spalte zurück. Der Index des Ergebnisses entspricht den Spalten des ursprünglichen DataFrame. 
# Spalten mit gemischten Typen werden mit dem Objekt dtype gespeichert.
df.dtypes


# In[8]:


plt.figure(figsize=(15,8))
plt.hist(df['sellingprice'],bins=200,color='#dc2f02')
plt.title('Verteilung des Verkaufspreises',fontsize=15)
plt.xticks(np.arange(0,df['sellingprice'].max(),15000))
plt.xlabel('Verkaufspreis',fontsize=12)
plt.ylabel('Freq',fontsize=12)


# In[9]:


df['sellingprice'].describe()


# Die Verteilung scheint schräg nach rechts zu verlaufen, wobei der Großteil der Verkaufspreise zwischen 15000 und 30000 liegt. Es gibt Ausreißer > 60000. 75 % der Daten haben einen Verkaufspreis < 200K. Ich gehe davon aus, dass es sich bei den teuersten Marken um Oldtimer oder High-End-Modelle wie Ferrari, BMW usw. handeln sollte. Das wird vielleicht klar, wenn wir dies mit der Marke vergleichen. Werfen wir einen groben Blick auf die Top-End-Marken

# In[10]:


df.loc[df['sellingprice']>60000,'make'].value_counts()


# Unsere Vermutung war richtig. Die Liste wird von BMW, Benz, Jaguar und Ferrari dominiert. Ein weiterer interessanter Punkt bei unserer Analyse ist die Tatsache, dass die Spalte "Marke" möglicherweise etwas bereinigt werden muss, da sich einige Marken zu wiederholen scheinen (BMW, bmw, Land Rover, land rover usw.). Der Einfachheit halber werden wir dies nicht tun und uns stattdessen auf die Gesamtanalyse konzentrieren

# In[11]:


df['make'].value_counts()[:10]


# In[12]:


top_make=df['make'].value_counts()[:10].index


# In[13]:


top_make


# Da wir fast 96 Marken haben, betrachten wir nur die Top 10 und veranschaulichen ihre Verkaufspreisentwicklung.
# 

# In[14]:


fig,ax=plt.subplots(5,2,figsize=(14,20))
color_list=['#0a9396','#ca6702','#ae2012','#9b2226','#001219','#005f73','#94d2bd','#e9d8a6','#e5e5e5','#e07a5f'] #coolors.co
i=0
for t in top_make:
    df.loc[df['make']==t,'sellingprice'].hist(ax=ax[i%5][i//5],bins=100,color=np.random.choice(color_list,replace=False))
    ax[i%5][i//5].set_xlabel('Verkaufspreis',fontsize=10)
    ax[i%5][i//5].set_ylabel('Frequency',fontsize=10)
    ax[i%5][i//5].set_title(f'Verteilung des Kaufpreises für {t}',fontsize=15)
    plt.subplots_adjust(hspace=0.45)
    i+=1


# In[15]:


df.loc[df['make']=='Ford','sellingprice'].describe()


# 
# - Die Verteilung der Top-10-Marken ist einzigartig und die Preisspanne ist unterschiedlich.
# -  Von diesen scheinen Honda und Chrysler eine größere Bandbreite bei der Verteilung der Verkaufspreise zu haben.
# - Der Verkaufspreis von Ford liegt eng beieinander und ist auf 50000 begrenzt. Es gibt einen Ausreißer, der in der Preisspanne oberhalb von 200.000 zu liegen scheint.
# - Fast alle der 10 Marken haben Spitzenwerte um 10000. Es gibt nur wenige Marken, die bimodale Spitzenwerte aufweisen.
# 
# Eine ähnliche Analyse wie die obige kann für verschiedene Karosserietypen durchgeführt werden.

# In[16]:


(df['body'].value_counts()[:10]/df.shape[0])*100



# Da 11 % der Gesamtdaten in dieser Spalte Null sind, werde ich sie in der Analyse entfernen.

# In[17]:


trans_df=df.loc[~(df['transmission'].isna()),]
trans_df.isna().sum() ## Bestätigt, dass Nullwerte entfernt werden.


# In[18]:


trans_df['transmission'].value_counts()


# In[19]:


plt.figure(figsize=(10,11))
# plt.hist(trans_df.loc[trans_df['transmission']=='automatic','sellingprice'],color='#b5179e',alpha=0.8,label='automatic',bins=100)
# plt.hist(trans_df.loc[trans_df['transmission']=='manual','sellingprice'],color='#480ca8',alpha=0.8,label='manual',bins=100)
# plt.legend()
sns.boxplot(x='transmission',y='sellingprice',data=trans_df,palette=['#b5179e','#480ca8'])
plt.title('Verteilung des Verkaufspreises',fontsize=15)
plt.xlabel('Selling Price',fontsize=10)
plt.ylabel('Freq',fontsize=10)


# Es besteht ein deutlicher Unterschied zwischen dem Getriebe und dem Verkaufstyp. Bei Fahrzeugen mit Automatikgetriebe scheint das Vorhandensein von Ausreißern eine große Rolle zu spielen. Der Median des Verkaufspreises von Fahrzeugen mit Automatikgetriebe ist höher als der von Fahrzeugen mit Schaltgetriebe.

# Da wir nun die Verteilung der Verkaufspreise kennen, können wir prüfen, wie sie sich zu den Kilometerständen verhalten. Im Allgemeinen würde ich erwarten, dass der Wiederverkaufswert umso geringer ist, je höher die Nutzung (Kilometerstand) ist. 
# 

# In[20]:


plt.figure(figsize=(10,10))
g=sns.scatterplot(x='odometer',y='sellingprice',data=df,color='#0d3b66',alpha=0.8)
g.set_title('Odometer vS Selling Price Correlation',fontsize=12)
g.set_xlabel('Odometer',fontsize=10)
g.set_ylabel('Selling Price',fontsize=10)
xlabels=['{:,.2f}'.format(x)+'k' for x in g.get_xticks()/10e3]
ylabels=['{:,.2f}'.format(y)+'k' for y in g.get_yticks()/10e3]
g.set_xticklabels(xlabels);
g.set_yticklabels(ylabels);


# Es zeigt sich, dass der Anstieg der Kilometerstände (> 45T) zwar den Verkaufspreis der Autos gesenkt hat, dass es aber auch Autos gab, die einen niedrigeren Kilometerstand aufwiesen, deren Verkaufspreis jedoch bei > 5k lag. Wir können also davon ausgehen, dass nicht nur der Kilometerstand, sondern auch andere Faktoren wie Marke, Modell, Bundesland usw. für den Verkaufspreis von Autos ausschlaggebend sind.

# # Marke & Model
# 
# Um diese Analyse durchzuführen, müssen die Nullwerte für Marke und Modell entfernt werden.
# 

# In[21]:


mod_df=df.dropna(axis=0,subset=['make','model'])
mod_df.isna().sum()


# In[22]:


ma_mo=list(zip(mod_df['make'],mod_df['model']))


# In[23]:


Counter(i for i in ma_mo).most_common()[:10]


# Ford scheint die Liste der meistverkauften Autos eindeutig zu dominieren. Nissan, Chevrolet, Honda und BMW sind weitere Marken.
# 

# In[24]:


plt.figure(figsize=(10,8))
plt.hist(df['condition'],bins=30,color='#023047')
plt.title('Verteilung des Zustands',fontsize=20)
plt.xlabel('Zustand',fontsize=15)
plt.ylabel('Freq',fontsize=15)


# Der Zustand der zu versteigernden Fahrzeuge wird zwischen 1 und 5 eingestuft. Es gibt keine eindeutige Verteilung, die sich aus dem Diagramm ableiten ließe. 45000 Autos sind in sehr gutem Zustand, und es gibt auch Autos mit den Bewertungen 1,8 und 3,5 in großer Zahl.
# 
# Lässt man die Spalte weg und ermittelt den durchschnittlichen Verkaufspreis entsteht folgendes :

# In[25]:


df['condition_bin'],bins=pd.cut(df['condition'],bins=4,retbins=True)


# In[26]:


plt.figure(figsize=(10,8))
sns.violinplot(y=df['condition_bin'],x=df['sellingprice'],palette=['#606c38','#283618','#dda15e','#bc6c25'])
plt.title('Zustand des Autos Vs Verkaufspreis',fontsize=12)
plt.ylabel('Zustand',fontsize=12)
plt.xlabel('Verkaufspreis',fontsize=12)


# Es ist deutlich zu erkennen, dass der Medianwert des Verkaufspreises mit zunehmendem Zustand der Fahrzeuge ansteigt. Wie aus den Diagrammen ersichtlich ist, werden die Bedingungen von vielen Ausreißern dominiert.

# In[27]:


df['year'].min(),df['year'].max()


# Wir haben Daten von 1982 bis 2015.
# 
# 
# Es gibt auch eine Spalte - saledate, die einige gute Informationen enthält. Bereinigen wir sie, um Wochentag, Monat und Tag zu extrahieren.

# In[28]:


df['sale_dow']=df['saledate'].apply(lambda x:re.search('^(\w+)\s',x).group(1))
df['sale_month']=df['saledate'].apply(lambda x:re.search('(\w+)\s(\d+)',x).group(1))
df['sale_day']=df['saledate'].apply(lambda x:re.search('(\w+)\s(\d+)',x).group(2))
df['sale_year']=df['saledate'].apply(lambda x:re.search('(\w+)\s(\d{4})',x).group(2))
df['sale_date']=df['saledate'].apply(lambda x:re.search('(\w+\s\d{2}\s\d{4})',x).group(1))
df['sale_date']=pd.to_datetime(df['sale_date'],format='%b %d %Y')


# In[29]:


df['sale_dow'].value_counts()



# In[30]:


df['sale_month'].value_counts()


# In[31]:


df['sale_day'].value_counts()


# In[32]:


df['sale_year'].value_counts()


# 
# Wir haben bereits gesehen, dass die Spalte Jahr Werte von 1982 bis 2015 enthält, während das Verkaufsjahr 2014, 2015 ist. Daraus können wir schließen, dass die Jahresspalte das Modelljahr ist.
# Welche Marken waren bei der Auktion am begehrtesten?
# 
# Schauen wir uns an, welches Jahr der hergestellten Marke den höchsten Verkaufswert hatte. Dann verstehen wir die Modelle in diesem Spitzenjahr.

# In[33]:


sale_model=df.groupby('year')['make'].count().sort_values(ascending=False)[:10].reset_index().rename(columns={'make':'total_units'})


# In[34]:


plt.figure(figsize=(10,8))
sns.barplot(x='year',y='total_units',data=sale_model,palette=sns.set_palette('Set1'))
plt.title('Anzahl der Verauften Autos pro Jahr',fontsize=15)
plt.xlabel('Jahr',fontsize=10)
plt.ylabel('Gesamtanzahl ',fontsize=10)


# 
# Aus dem Diagramm geht hervor, dass Autos der Marke 2012 die meisten Auktionen hatten, gefolgt von 2013 und 2014. Oldtimer verkauften sich also nicht so gut, und die Leute interessierten sich mehr für die neuesten Marken.Interessant!
# 
# Mal sehen, welches Modell beim durchschnittlichen Verkaufspreis an der Spitze steht.
# 

# In[35]:


sale_year=df.groupby('year')['sellingprice'].mean().sort_values(ascending=False)[:10].reset_index()


# In[36]:


plt.figure(figsize=(8,8))
sns.barplot(x='year',y='sellingprice',data=sale_year)
plt.title('Top 10 Modelle (Verkaufspreis)')
plt.xlabel('Jahr')
plt.ylabel('Verkaufspreis')


# Das Modell 2012 mag die höchsten Auktionen nach Gesamteinheiten erzielt haben, aber wenn es um den durchschnittlichen Verkaufspreis geht, liegt das Modell 2015 an der Spitze, gefolgt vom Modell 2014.Eine weitere interessante Sache ist der Preis des Modells 1982, das die Top 10 der Verkaufspreise dominiert.
# 
# Die bloße Betrachtung des Modelljahrs könnte keine wertvollen Informationen liefern, da es mehrere Marken für das Modelljahr gibt. Lassen Sie uns das Jahr und die Markenspalte zusammenfassen und diese Analyse erneut betrachten.
# 

# In[37]:


df['year_make']=df['year'].astype('str')+'_'+df['make']


# In[38]:


df['year_make'].value_counts().reset_index().rename(columns={'year_make':'units','index':'year_make'})[:10]


# In[39]:


sale_make=df.groupby('year_make')['sellingprice'].mean().sort_values(ascending=False)[:10].reset_index()
units_make=df['year_make'].value_counts().reset_index().rename(columns={'year_make':'units','index':'year_make'})[:10]


# In[40]:


plt.figure(figsize=(22,12))

plt.subplot(1,2,1)

a=sns.barplot(x='sellingprice',y='year_make',data=sale_make)
a.set_title('Top 10 Marke&Variante nach Verkaufspreis(Avg)',fontsize=15)
a.set_ylabel('Jahr & Marke',fontsize=12)
a.set_xlabel('Verkaufspreis',fontsize=12)

plt.subplot(1,2,2)
b=sns.barplot(x='units',y='year_make',data=units_make)
b.set_title('Top 10 Marke&Variante nach verkauften Einheiten',fontsize=15)
b.set_ylabel('Jahr & Marke',fontsize=12)
b.set_xlabel('Verkaufte Einheiten',fontsize=12)


plt.subplots_adjust(hspace=0.45)


# 
# - Wenn es um Modelle mit einem höheren Verkaufspreis geht, dominieren Ferraris, Rolls Royce und Bentleys die Liste. Nicht überraschend.
# - Bei den insgesamt verkauften Einheiten dominieren Nissan, Ford, Hyundai und Chevrolet die Liste.
# 
# 
# # Schlussfolgerung:
# 
# In dieser Analyse habe ich eine explorative Analyse des Autoauktionsdatensatzes durchgeführt - kurz gesagt habe ich gesehen, wie die Verteilung des Verkaufspreises aussieht, wie Typ und Modell der verkauften Autos aussehen und wie jeder der Parameter wie Kilometerstand und Getriebe den Wert des Verkaufspreises beeinflusst.
# 
