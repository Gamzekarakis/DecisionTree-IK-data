#!/usr/bin/env python
# coding: utf-8

# ### Decision Tree Classification

# * Veri setimizde IK departmanının değerlendirmesi öncesi hangi  CV 'lerin IK değerlendirmesine düşmesi gerektiğini belirleyen çalışma yapacağız .

# In[1]:


import numpy as np
import pandas as pd
from sklearn import tree


# In[2]:


df=pd.read_csv("DecisionTreesClassificationDataSet.csv")


# In[3]:


df.head()


# * scikit-learn kütüphanesi decision tree'lerin düzgün çalışması için herşeyin rakamsal olmasını bekliyor bu nedenle veri setimizdeki tüm Y ve N değerlerini 0 ve 1 olarak düzeltiyoruz. Aynı sebeple eğitim seviyesini de BS:0 MS:1 ve PhD:2 olarak güncelliyoruz. map() kullanarak boş hücreler veya geçersiz değer girilen hücreler NaN ile doldurulacaktır, buna şuandaki veri setimizde ihtiyac yok

# In[4]:


duzetme_mapping = {'Y': 1, 'N': 0}

df['IseAlindi'] = df['IseAlindi'].map(duzetme_mapping)
df['SuanCalisiyor?'] = df['SuanCalisiyor?'].map(duzetme_mapping)
df['Top10 Universite?'] = df['Top10 Universite?'].map(duzetme_mapping)
df['StajBizdeYaptimi?'] = df['StajBizdeYaptimi?'].map(duzetme_mapping)
duzetme_mapping_egitim = {'BS': 0, 'MS': 1, 'PhD': 2}
df['Egitim Seviyesi'] = df['Egitim Seviyesi'].map(duzetme_mapping_egitim)
df.head()


# In[5]:


# Sonuç sütununu ayırıyoruz


# In[6]:


y = df['IseAlindi']
X = df.drop(['IseAlindi'], axis=1)


# In[7]:


X.head()


# In[8]:


# Decision Tree'mizi oluşturuyoruz:



clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,y)


# In[10]:


# Prediction yapalım şimdi
# 5 yıl deneyimli, hazlihazırda bir yerde çalışan ve 3 eski şirkette çalışmış olan, eğitim seviyesi Lisans
# top-tier-school mezunu değil
print (clf.predict([[5, 1, 3, 0, 0, 0]]))


# In[11]:


# Toplam 2 yıllık iş deneyimi, 7 kez iş değiştirmiş çok iyi bir okul mezunu şuan çalışmıyor
print (clf.predict([[2, 0, 7, 0, 1, 0]]))


# In[ ]:




