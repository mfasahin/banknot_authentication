import os 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#load dataset
df = pd.read_csv('data_banknote_authentication.csv', sep=';')
#print(df.head())
#print(df.tail())

# variance;skewness;curtosis;entropy;class
df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'y']

#object -> categorical
"""
df.variance = pd.Categorical(df.variance)
df.skewness = pd.Categorical(df.skewness) 
df.curtosis = pd.Categorical(df.curtosis)
df.entropy = pd.Categorical(df.entropy)
df.y = pd.Categorical(df['y'])
#print(df.info())
"""

for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        #print(f"Converted {col} to numeric.")

#print(df.head())
#print(df.info())

#eksik veri analizi
""""
eksik_veri_var_mi = df.isnull().values.any()
print(f"Eksik veri var mı? {eksik_veri_var_mi}")

toplam_eksik_veri = df.isnull().sum()
print("Toplam eksik veri sayısı:\n", toplam_eksik_veri)

print("temizlik öncesi satır sayısı:", len(df))
"""

df = df.dropna() # verileri siliyor ama kalıcı değil
#print("temizlik sonrası satır sayısı:", len(df))
eksik_veri_var_mi = df.isnull().values.any()


print(f"Eksik veri var mı? {eksik_veri_var_mi}")

#veri görselleştirme

# Pairplot
#sns.pairplot(df, hue='y', palette='husl')
#plt.show()

# Korelasyon haritasi
#sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
#plt.show()

