import  pandas as pd
import  numpy as np
import  matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from  sklearn.linear_model import LogisticRegression
from  sklearn.ensemble import RandomForestClassifier
from  sklearn.ensemble import  GradientBoostingClassifier
import pickle
from pathlib import Path

df_load = pd.read_csv('https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/dqlab_telco_final.csv')
print(df_load.shape)
print( df_load.head())
print(df_load.customerID.unique())

#grafik churn
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
labels= (['Yes','No'])
churn = df_load.Churn.value_counts()
ax.pie(churn, labels=labels, autopct= '%.0f%%')
plt.show()
