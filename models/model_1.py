"""
Authors:
    Pratik Goyal (B18BB024)     Piyush Mathur (B18BB021)
    goyal.9@iitj.ac.in          mathur.2@iitj.ac.in
    
Supervisor:
    Dr. Pankaj Yadav
    pyadav@iitj.ac.in
    
Description:
    Statistical model to predict the risk of diabetes in which Logistic Regression
    model is used and no data has been removed from the dataset even if they contained
    'NULL' or '0' values.

Steps:
    Loading Data
    Checking the Data integrity
    Checking for Correlations
    Data Visualization
    Feature Selection
    Feeding data into LR model
    Result: without feature selection
    Result: with feature selection
"""



# Importing libraries
import pandas as pd # data processing csv file I/O
import numpy as np # high-performance multidimensional array object and tools for working with these arrays
import matplotlib.pyplot as plt # plotting graphs
import seaborn as sns # statistical graphs
from sklearn.model_selection import train_test_split # from scikit learn library Split arrays or matrices into random train and test subsets 
from sklearn.linear_model import LogisticRegression # inbuilt functions for logistic regression
from sklearn.feature_selection import RFE # for feature selection
from sklearn import metrics 



# Created a boundary function for clear visualization on console
def boundary():
    print("\n" + "--"*35 + "\n")



# Load data
dataset = pd.read_csv("../dataset/pimadata.csv")  #pimadata/germandata/combined_data



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
    print(' (' + '%.2f'%((count*100)/len(dataset)) + '%)')
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
print("\nDependent Variables are:")
print(list(feature_names))
print("\nIndependent variable is:")
print("'Outcome'")
boundary()



# SELECTING GOOD FEATURES
print("The goal of Recursive Feature Elimination (RFE) is to select features by feature ranking with recursive feature elimination.\n")
high_score = 0    # Variable which will store the highest score value
nof = 0           # Variable which will store the the total number of optimum features
score_list = []   # list for storing the model scores
# Splitting dataset into training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state = 1)
# used model
model = LogisticRegression(solver='lbfgs', multi_class='auto',max_iter=1000)
# applying RFE
for feature_count in range(1, len(feature_names)+1):
    rfe = RFE(model, feature_count)
    x_train_rfe = rfe.fit_transform(x_train, y_train)
    x_test_rfe = rfe.transform(x_test)
    model.fit(x_train_rfe, y_train)
    score = model.score(x_test_rfe, y_test) #individual score
    score_list.append(score)  #appending individual score in score_list array
    if(score > high_score): 
        high_score = score   #updating the high score if score > high_score
        nof = feature_count  #updating the value of no of features selected
#plotting model score v/s number of features selected
plt.figure()
plt.title('RFE with Logistic Regression')
plt.xlabel('Number of features selected')
plt.ylabel('Model Score')
plt.plot(range(1, len(feature_names)+1), score_list)        
plt.show()
print("RFE with Logistic Regression curve has been plotted\n")
print('RFE was used to find the optimum number of features\n')
print("Total optimum features that are selected: %d" %nof)
print("Score with these %d features: %f" % (nof, high_score))
boundary()

# we now know the optimum number of features 
# Applying RFE for those optimum features
rfe = RFE(model, nof)          #initializing RFE
x_rfe = rfe.fit_transform(x, y) #transforming data using RFE
model.fit(x_rfe, y)             #fitting the data to the model
table = pd.Series(rfe.support_, feature_names) #variable to store True/False table for corresponding features
selected_features_rfe = table[table == True].index  #variable to store selected features
discarded_features_rfe = table[table == False].index  #variable to store discarded features
print('The True/False table for corresponding feature after RFE is :')
print(table)
print('\nThe selected features are :')
print(list(selected_features_rfe))
print('\nThe discarded features are :')
print(list(discarded_features_rfe))
boundary()



# Final Model output without RFE
print('The final report of the model when the result of RFE is excluded\n')

# Independent and dependent variables
x1 = dataset[feature_names]  #independent variables
y1 = dataset['Outcome']      #dependent variable

# Splitting dataset into training set and test set
x_train1, x_test1, y_train1, y_test1 = train_test_split(x1, y1, test_size=0.25, random_state = 1)

# Applying Logistic Regression
model = LogisticRegression(random_state = 0, max_iter=1000)
model.fit(x_train1, y_train1)

# Predicting Outcomes
y_pred1 = model.predict(x_test)

# Making confusion matrix
cnf_matrix1 = metrics.confusion_matrix(y_test1, y_pred1)
print("\nThe confsuion matrix is:")
print(cnf_matrix1)

# Making report matrix
print("\nThe report matrix is:")
print(metrics.classification_report(y_test1, y_pred1))
boundary()



# Final Model output with RFE
print('The final report of the model when the result of RFE is considered\n')

# Independent and dependent variables
x2 = dataset[selected_features_rfe]  #independent variables
y2 = dataset['Outcome']      #dependent variable

# Splitting dataset into training set and test set
x_train2, x_test2, y_train2, y_test2 = train_test_split(x2, y2, test_size=0.25, random_state = 1)

# Applying Logistic Regression
model = LogisticRegression(random_state = 0, max_iter=1000)
model.fit(x_train2, y_train2)

# Predicting Outcomes
y_pred2 = model.predict(x_test2)

# Making confusion matrix
cnf_matrix2 = metrics.confusion_matrix(y_test2, y_pred2)
print("\nThe confsuion matrix is:")
print(cnf_matrix2)

# Making report matrix
print("\nThe report matrix is:")
print(metrics.classification_report(y_test2, y_pred2))
boundary()

