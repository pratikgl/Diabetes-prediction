# Importing libraries
import pandas as pd # data processing csv file I/O
import numpy as np # high-performance multidimensional array object and tools for working with these arrays
import matplotlib.pyplot as plt # plotting graphs
import math # mathematical functions
import seaborn as sns # statistical graphs
from sklearn.model_selection import train_test_split # from scikit learn library Split arrays or matrices into random train and test subsets 
from sklearn.linear_model import LogisticRegression # inbuilt functions for logistic regression
from sklearn import metrics 



# Load data
pdata = pd.read_csv("pimadata.csv")


# Data description
"""
Pregnancies: Number of times pregnant
Glucose: Plasma glucose concentration over 2 hours in an oral glucose tolerance test
BloodPressure: Diastolic blood pressure (mm Hg)
SkinThickness: Triceps skin fold thickness (mm)
Insulin: 2-Hour serum insulin (mu U/ml)
BMI: Body mass index (weight in kg/(height in m)2)
DiabetesPedigreeFunction: Diabetes pedigree function (a function which scores likelihood of diabetes based on family history)
Age: Age (years)
Outcome: Class variable (0 if non-diabetic, 1 if diabetic)
"""


# Check for data
print("\nThe following dataset has been used:")
print(pdata)


# Check for null values
print("\nNULL values in the dataset are:")
print(pdata.isnull().sum())


# Correlation between independent variables and outcomes
print("\nThe correlation matrix is:")
correlations = pdata.corr()
print(correlations['Outcome'].sort_values(ascending=False))


# Diabetic population VS non diabtic population
print("\nComparing non diabetic = 0 vs diabetic = 1")
pdata.Outcome.value_counts()
plt.title('Figure-1')
sns.countplot(x = "Outcome", data = pdata)
plt.show()
print("Output = figure-1")


# Dividing inedpendent and dependent (outcome) variables
x = pdata[['Pregnancies', 'Glucose', 'BloodPressure',  'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = pdata['Outcome']


# Splitting dataset into training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state = 0)


# Applying Logistic Regression
model = LogisticRegression(random_state = 0, max_iter=1000)
model.fit(x_train, y_train)


# Predicting Outcomes
y_pred = model.predict(x_test)


# Making confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print("\nThe confsuion matrix is:")
print(cnf_matrix)


# Making report matrix
print("\nThe report matrix is:")
print(metrics.classification_report(y_test, y_pred))


# Refrences:
"""
Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011
towardsdatascience.com/real-world-implementation-of-logistic-regression-5136cefb8125

"""