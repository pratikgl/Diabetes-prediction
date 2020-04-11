# Importing libraries
from sklearn.datasets import load_boston
import pandas as pd # data processing csv file I/O
import numpy as np # high-performance multidimensional array object and tools for working with these arrays
import matplotlib.pyplot as plt # plotting graphs
import math # mathematical functions
import seaborn as sns # statistical graphs
import statsmodels.api as sm
from sklearn.model_selection import train_test_split # from scikit learn library Split arrays or matrices into random train and test subsets
from sklearn.linear_model import LogisticRegression # inbuilt functions for logistic regression
from sklearn.feature_selection import RFE
from sklearn import metrics

# Load data
pdata = pd.read_csv("../dataset/pimadata.csv")


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

#RFE starts

#finding optimum no. of features

#no. of features
nof_list=np.arange(1,8)
high_score=0

#Variable to store the optimum features
nof=0           
score_list =[]
for n in range(len(nof_list)):
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state = 1)
    model = LogisticRegression(random_state=40, max_iter=1000)
    rfe = RFE(model,nof_list[n])
    x_train_rfe = rfe.fit_transform(x_train,y_train)
    x_test_rfe = rfe.transform(x_test)
    model.fit(x_train_rfe,y_train)
    score = model.score(x_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
        
print('\n'+'RFE was used to find the optimum number of features')
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))

#now we got that 5 is the optimum number of features

cols = list(x.columns)
model = LogisticRegression(random_state=40)

#Initializing RFE model
rfe = RFE(model, 5)             
#Transforming data using RFE
x_rfe = rfe.fit_transform(x,y)  
#Fitting the data to model
model.fit(x_rfe,y)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)

#'SkinThickness', 'Insulin', 'Age' were eliminated

#RFE ends

x2 = pdata[['Pregnancies', 'Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigreeFunction']]

# Splitting dataset into training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state = 1)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y, test_size=0.25, random_state = 1)
# Applying Logistic Regression
model=LogisticRegression(random_state=40)
model2=LogisticRegression(random_state=40)
model2.fit(x2_train, y2_train)
model.fit(x_train, y_train)


# Predicting Outcomes
y_pred = model.predict(x_test)
y2_pred = model2.predict(x2_test)


# Making confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf2_matrix = metrics.confusion_matrix(y2_test, y2_pred)

print('\n'+'THE MODEL BEFORE FEATURE SELECTION SHOWED FOLLOWING RESULTS'+'\n')

print(cnf_matrix)

print(metrics.classification_report(y_test,y_pred))

print('\n'+'THE MODEL AFTER FEATURE SELECTION SHOWED FOLLOWING RESULTS'+'\n')

print(cnf2_matrix)

print(metrics.classification_report(y2_test,y2_pred))




# Refrences:
"""
Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011

"""