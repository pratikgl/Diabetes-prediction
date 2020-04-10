"""
Description
"""


# Importing libraries
import pandas as pd # data processing csv file I/O
import numpy as np # high-performance multidimensional array object and tools for working with these arrays
import matplotlib.pyplot as plt # plotting graphs
import math # mathematical functions
import seaborn as sns # statistical graphs
from sklearn.model_selection import train_test_split # from scikit learn library Split arrays or matrices into random train and test subsets 
from sklearn.linear_model import LogisticRegression # inbuilt functions for logistic regression
from sklearn import metrics 



# Created a boundary function for clear visualization on console
def boundary():
    print("\n" + "--"*35 + "\n")


# Load data
dataset = pd.read_csv("../dataset/pimadata.csv")


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
boundary()
print("The following dataset has been used:\n")
print(dataset)
boundary()
print("Information regarding all the columns :\n")
print(dataset.info())
boundary()


# Diabetic population VS non diabtic population
print("Comparing non diabetic = 0 vs diabetic = 1 :")
print(dataset.groupby('Outcome').size())
boundary()


# Check for null values
print("NULL values in the dataset are:")
print(dataset.isnull().sum())
boundary()


# Check for 0 values
print("Number of 0-entries in the dataset are:\n")
for feature_names in dataset.columns[0:8]:
    print('Number of 0-entries for "' + feature_names + '" feature: ' , end="")
    count = (np.count_nonzero(dataset[feature_names] == 0))
    print(count, end=" ")
    print(' (' + '%.2f'%((count*100)/768) + '%)')
boundary()


# Checking the Correlations between variables
print("The correlation of variables with Outcome is:")
correlations = dataset.corr()
print(correlations['Outcome'].sort_values(ascending=False))


# Correlation Matrix
print("\nCorrelation Marix has been printed")
sns.heatmap(
    data=dataset.corr(), 
    annot=True, #printing values on cells
    fmt='.2f', #rounding off
    cmap='RdYlGn' #colors
)
plt.title("Correlation Matrix")
fig = plt.gcf()
fig.set_size_inches(10, 8)
plt.show()
boundary()


# Visualising Data
plot = sns.pairplot(data=dataset, hue='Outcome')
plot.fig.suptitle("Before removing zero", y=1.02)
plt.show()
print("Visualization has been plotted")
boundary()


# Dividing inedpendent and dependent (outcome) variables
feature_names = dataset.columns[0:8]
x = dataset[feature_names]
y = dataset['Outcome']
print("Dependent Variables are:")
print(feature_names)
print("\nIndependent variable is:")
print("'Outcome'")
boundary()



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