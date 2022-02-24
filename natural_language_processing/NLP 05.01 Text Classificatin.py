# Databricks notebook source
# MAGIC %md
# MAGIC ### Setup

# COMMAND ----------

# import sys
# sys.path.append("/Workspace/Repos/bladnman@gmail.com/databricks_tools")
# from brick_utils.SparkUtils_01 import SparkUtils
# su = SparkUtils(spark)
from brick_utils.SparkUtils_01 import SparkUtils
su = SparkUtils(spark)

# COMMAND ----------

# MAGIC %md
# MAGIC # Coursework

# COMMAND ----------

# MAGIC %md
# MAGIC ## Learning Scikit-Learn

# COMMAND ----------

import numpy as np
import pandas as pd

# COMMAND ----------

df = pd.read_csv('data/smsspamcollection.tsv', sep="\t")
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC We are going to use the `length` and `punct` for now

# COMMAND ----------

df.isnull().sum()

# COMMAND ----------

len(df)

# COMMAND ----------

df['label'].unique()

# COMMAND ----------

df['label'].value_counts()

# COMMAND ----------

import matplotlib.pyplot as plt

plt.xscale('log')
bins = 1.15**(np.arange(0,50))
plt.hist(df[df['label'] == 'ham']['length'], bins=bins, alpha=0.8)
plt.hist(df[df['label'] == 'spam']['length'], bins=bins, alpha=0.8)
plt.legend(('ham','spam'))
plt.show()

# COMMAND ----------

plt.xscale('log')
bins = 1.5**(np.arange(0,15))
plt.hist(df[df['label'] == 'ham']['punct'], bins=bins, alpha=0.8)
plt.hist(df[df['label'] == 'spam']['punct'], bins=bins, alpha=0.8)
plt.legend(('ham','spam'))
plt.show()

# COMMAND ----------

from sklearn.model_selection import train_test_split

# COMMAND ----------


X = df[['length','punct']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)

# COMMAND ----------

X_train.shape

# COMMAND ----------

X_test.shape

# COMMAND ----------

y_test.shape

# COMMAND ----------

from sklearn.linear_model import LogisticRegression

# COMMAND ----------

lr_model = LogisticRegression(solver='lbfgs')

# COMMAND ----------

lr_model.fit(X_train,y_train)

# COMMAND ----------

from sklearn import metrics

# COMMAND ----------

predictions = lr_model.predict(X_test)

# COMMAND ----------

metrics.confusion_matrix(y_test, predictions)

# COMMAND ----------

# You can make the confusion matrix less confusing by adding labels:
df = pd.DataFrame(metrics.confusion_matrix(y_test,predictions), index=['ham','spam'], columns=['ham','spam'])
df

# COMMAND ----------

metrics.classification_report(y_test, predictions)

# COMMAND ----------

metrics.accuracy_score(y_test, predictions)

# COMMAND ----------

# Print a classification report
print(metrics.classification_report(y_test,predictions))

# COMMAND ----------


