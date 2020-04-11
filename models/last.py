"""
Description
"""


# Importing libraries
import pandas as pd # data processing csv file I/O
import numpy as np # high-performance multidimensional array object and tools for working with these arrays
import matplotlib.pyplot as plt # plotting graphs
import operator
import seaborn as sns # statistical graphs
from sklearn.model_selection import train_test_split # from scikit learn library Split arrays or matrices into random train and test subsets 
from sklearn.linear_model import LogisticRegression # inbuilt functions for logistic regression
from sklearn import metrics 
# new library added
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score


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


# Removing 0 values
print("Some features contain 0, it doesn't make sense here and this indicates missing value therefore we will remove all those columns which has 0-entries\n")
print("Removing 0-entries from Glucose, BloodPressure, SkinThickness, Insulin, BMI\n")
ds_0_remove = dataset[
    (dataset.Glucose != 0)
    & (dataset.BloodPressure != 0)
    & (dataset.SkinThickness != 0)
    & (dataset.Insulin != 0) 
    & (dataset.BMI != 0)]

print("After deleting 0-entries the final size of the data is:")
print(ds_0_remove.shape)
print("\nComparing non diabetic = 0 vs diabetic = 1 :")
print(ds_0_remove.groupby('Outcome').size())
boundary()


# Checking the Correlations between variables
#Before Removing Zero
print("Before removing 0")
print("The correlation of variables with Outcome is:")
correlations = dataset.corr()
print(correlations['Outcome'].sort_values(ascending=False))


# Correlation Matrix
# Before removing zero
print("\nCorrelation Marix has been printed")
sns.heatmap(
    data=dataset.corr(), #feeding data
    annot=True, #printing values on cells
    fmt='.2f', #rounding off
    cmap='RdYlGn' #colors
)

plt.title("Correlation Matrix before removing Zero")
fig = plt.gcf()
fig.set_size_inches(10, 8)
plt.show()

boundary()


# Checking the Correlations between variables
# After Removing Zero
print("After removing 0")
print("The correlation of variables with Outcome is:")
correlations = ds_0_remove.corr()
print(correlations['Outcome'].sort_values(ascending=False))


# Correlation Matrix
# After removing zero
print("\nCorrelation Marix has been printed")
sns.heatmap(
    data=ds_0_remove.corr(), #feeding data
    annot=True, #printing values on cells
    fmt='.2f', #rounding off
    cmap='RdYlGn' #colors
)

plt.title("Correlation Matrix after removing Zero")
fig = plt.gcf()
fig.set_size_inches(10, 8)
plt.show()

boundary()


# Comparison of the Visualisation of the data
plot = sns.pairplot(data=dataset, hue='Outcome')
plot.fig.suptitle("Before removing zero", y=1.02)
plt.show()
print("Visualization before removing zero has been plotted")
boundary()

plot = sns.pairplot(data=ds_0_remove, hue='Outcome')
plot.fig.suptitle("Before removing zero", y=1.02)
plt.show()
print("Visualization after removing zero has been plotted")
boundary()



# Dividing inedpendent and dependent (outcome) variables
feature_names = dataset.columns[0:8]
x = ds_0_remove[feature_names]
y = ds_0_remove['Outcome']
print("Dependent Variables are:")
print(feature_names)
print("\nIndependent variable is:")
print("'Outcome'")
boundary()


# Selecting good features
print("The goal of Recursive Feature Elimination (RFE) is to select features by feature ranking with recursive feature elimination.", end=" ")
print("For more confidence of features selection we used K-Fold Cross Validation with Stratified k-fold.\n")
strat_k_fold = StratifiedKFold(
    n_splits=10
)
logreg_model = LogisticRegression(max_iter=1000)
rfecv = RFECV(
    estimator=logreg_model,
    step=1,
    cv=strat_k_fold,
    scoring='accuracy'
)
rfecv.fit(x, y)
plt.figure()
plt.title('RFE with Logistic Regression')
plt.xlabel('Number of features selected')
plt.ylabel('Accuracy')
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
print("RFE with Logistic Regression curve has been plotted\n")
# grid_scores_ returns a list of accuracy scores for each of the features selected
print("grid_scores_ returns a list of accuracy scores for each of the features selected\n")
print("The rfecv grid_scores_ :")
print(rfecv.grid_scores_)
# support_ is another attribute to find out the features which contribute the most to predicting
print("\nThe rfecv.support_ : ")
print(rfecv.support_)
# Good Features
new_features = list(filter(
    lambda x: x[1],
    zip(feature_names, rfecv.support_)
))
new_features = list(map(operator.itemgetter(0), new_features))
print('\nThe most suitable features for prediction are :')
print(new_features)
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