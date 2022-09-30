# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization

# pip install seaborn
import seaborn as sns

import matplotlib.pyplot as plt 
# %matplotlib inline
# plt.show()

# machine learning
# pip install scikit-learn

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]

#print(train_df.columns.values)
#print(test_df.columns.values)

# preview the data
# train_df.head()
# print(train_df.head())
# print(test_df.head())

# print (train_df.tail())

#train_df.info()
#print('_'*40)
#test_df.info()
#print(combine)


print(train_df.describe())
# Review survived rate using `percentiles=[.61, .62]` knowing our problem description mentions 38% survival rate.
# Review Parch distribution using `percentiles=[.75, .8]`
# SibSp distribution `[.68, .69]`
# Age and Fare `[.1, .2, .3, .4, .5, .6, .7, .8, .9, .99]`
print (train_df.describe(include=['O']))