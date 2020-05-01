"""
Authors:
    Pratik Goyal (B18BB024)     Piyush Mathur (B18BB021)
    goyal.9@iitj.ac.in          mathur.2@iitj.ac.in
    
Supervisor:
    Dr. Pankaj Yadav
    pyadav@iitj.ac.in
    
Description:
    ROC curve is plotted for every model (RFE is considered)
"""



# Importing libraries
import pandas as pd # data processing csv file I/O
import numpy as np # high-performance multidimensional array object and tools for working with these arrays
import matplotlib.pyplot as plt # plotting graphs
from sklearn.model_selection import train_test_split # from scikit learn library Split arrays or matrices into random train and test subsets
from sklearn.linear_model import LogisticRegression # inbuilt functions for logistic regression
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.impute import KNNImputer


# Load data
dataset = pd.read_csv("../dataset/pimadata.csv")


# defining features
feature_names = dataset.columns[0:8]


# Removing 0 entries
ds_0_remove = dataset[
    (dataset.Glucose != 0)
    & (dataset.BloodPressure != 0)
    & (dataset.SkinThickness != 0)
    & (dataset.Insulin != 0) 
    & (dataset.BMI != 0)]



# DATA IMPUTATION USING MEDIAN BY TARGET
# Replacing 0 entries to NaN
replace_0_median = dataset.copy()     #variable for storing replaced values
replace_0_median[[                    #converting 0-entries to NaN
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI'
    ]] = replace_0_median[[
        'Glucose',
        'BloodPressure',
        'SkinThickness',
        'Insulin',
        'BMI'
        ]].replace(0,np.NaN)
# For replacing missing values we are using median by target (Outcome)
def median_target(var):   
    temp = replace_0_median[replace_0_median[var].notnull()]
    temp = temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].median().reset_index()
    return temp
# Final replacement with median values 
def replace(feature, ds):
    temp = median_target(feature)
    ds.loc[
        (ds['Outcome'] == 0 ) 
        & (ds[feature].isnull()), 
        feature
        ] = temp[feature][0]
    ds.loc[
        (ds['Outcome'] == 1 ) 
        & (ds[feature].isnull()), 
        feature
        ] = temp[feature][1]
    return 0

replace('Glucose', replace_0_median)       # Conversion for 'Glucose'
replace('BloodPressure', replace_0_median) # Conversion for 'BloodPressure'
replace('SkinThickness', replace_0_median) # Conversion for 'SkinThickness'
replace('Insulin', replace_0_median)       # Conversion for 'Insulin'
replace('BMI', replace_0_median)           # Conversion for 'BMI'



# DATA IMPUTATION USING KNN IMPUTATION METHOD
# Replacing 0 entries to NaN
replace_0_KNN = dataset.copy()     #variable for storing replaced values
replace_0_KNN[[                    #converting 0-entries to NaN
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI'
    ]] = replace_0_KNN[[
        'Glucose',
        'BloodPressure',
        'SkinThickness',
        'Insulin',
        'BMI'
        ]].replace(0,np.NaN)
# applying KNN imputer
imputer = KNNImputer(n_neighbors = 5)
# result in 2-d array
replace_0_KNN = imputer.fit_transform(replace_0_KNN)
# conversion o array into 2-d panda array
replace_0_KNN = pd.DataFrame(
    replace_0_KNN, 
    index=np.arange(0, len(replace_0_KNN)), 
    columns = [
        'Pregnancies', 
        'Glucose', 
        'BloodPressure', 
        'SkinThickness', 
        'Insulin', 
        'BMI', 
        'DiabetesPedigreeFunction', 
        'Age',
        'Outcome'])



def apply_rfe(x, y):
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
    
    # we now know the optimum number of features 
    # Applying RFE for those optimum features
    rfe = RFE(model, nof)          #initializing RFE
    x_rfe = rfe.fit_transform(x, y) #transforming data using RFE
    model.fit(x_rfe, y)             #fitting the data to the model
    table = pd.Series(rfe.support_, feature_names) #variable to store True/False table for corresponding features
    selected_features_rfe = table[table == True].index  #variable to store selected features
    return selected_features_rfe



# DIVIDING INDEPENDENT AND DEPENDENT (OUTCOME) VARIABLES
# Before removing 0-entries
x1 = dataset[feature_names]
y1 = dataset['Outcome']
final_features = apply_rfe(x1, y1)
x1 = dataset[final_features]

# After removing 0-entries
x2 = ds_0_remove[feature_names]
y2 = ds_0_remove['Outcome']
final_features = apply_rfe(x2, y2)
x2 = ds_0_remove[final_features]

# After imputing 0-entries using median by outcome
x3 = replace_0_median[feature_names]
y3 = replace_0_median['Outcome']
final_features = apply_rfe(x3, y3)
x3 = replace_0_median[final_features]

# After imputing 0-entries using KNN imputation
x4 = replace_0_KNN[feature_names]
y4 = replace_0_KNN['Outcome']
final_features = apply_rfe(x4, y4)
x4 = replace_0_KNN[final_features]




# Splitting dataset into training set and test set
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.25, random_state = 1)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.25, random_state = 1)
x3_train, x3_test, y3_train, y3_test = train_test_split(x3, y3, test_size=0.25, random_state = 1)
x4_train, x4_test, y4_train, y4_test = train_test_split(x4, y4, test_size=0.25, random_state = 1)


# Applying model
model = LogisticRegression(random_state=40,solver='lbfgs', multi_class='auto',max_iter=1000)

# Before removing 0-entries
model.fit(x1_train, y1_train)
lgreg_prob1 = model.predict_proba(x1_test)         # predict probabilities
lgreg_prob1 = lgreg_prob1[:, 1]   # keep probabilities for the positive outcome only
lgreg_auc1  = roc_auc_score(y1_test, lgreg_prob1)  # calculating ROC AUC score

# After removing 0-entries
model.fit(x2_train, y2_train)
lgreg_prob2 = model.predict_proba(x2_test)         # predict probabilities
lgreg_prob2 = lgreg_prob2[:, 1]   # keep probabilities for the positive outcome only
lgreg_auc2  = roc_auc_score(y2_test, lgreg_prob2)  # calculating ROC AUC score


# After imputing 0-entries using median by outcome
model.fit(x3_train, y3_train)
lgreg_prob3 = model.predict_proba(x3_test)         # predict probabilities
lgreg_prob3 = lgreg_prob3[:, 1]   # keep probabilities for the positive outcome only
lgreg_auc3  = roc_auc_score(y3_test, lgreg_prob3)  # calculating ROC AUC score


# After imputing 0-entries using KNN imputation
model.fit(x4_train, y4_train)
lgreg_prob4 = model.predict_proba(x4_test)         # predict probabilities
lgreg_prob4 = lgreg_prob4[:, 1]   # keep probabilities for the positive outcome only
lgreg_auc4  = roc_auc_score(y4_test, lgreg_prob4)  # calculating ROC AUC score

# No skill
no_skill_auc = roc_auc_score(y1_test, [0]*len(y1_test)) # calculating ROC AUC score

# summarize scores
print('\nWith no skills \nROC AUC = %.3f' % (no_skill_auc))
print('\nBefore removing 0-entries \nROC AUC = %.3f' % (lgreg_auc1))
print('\nAfter removing 0-entries  \nROC AUC = %.3f' % (lgreg_auc2))
print('\nAfter imputing 0-entries using median by target \nROC AUC = %.3f' % (lgreg_auc3))
print('\nAfter imputing 0-entries using KNN imputation \nROC AUC = %.3f' % (lgreg_auc4))



# fpr: false positive rate;   tpr: true positive rates
fpr0, tpr0, thresholds0 = roc_curve(y1_test, [0]*len(y1_test))
fpr1, tpr1, thresholds1 = roc_curve(y1_test, lgreg_prob1)
fpr2, tpr2, thresholds2 = roc_curve(y2_test, lgreg_prob2)
fpr3, tpr3, thresholds3 = roc_curve(y3_test, lgreg_prob3)
fpr4, tpr4, thresholds4 = roc_curve(y4_test, lgreg_prob4)


# plot the roc curve for the model
plt.figure(dpi = 600)
plt.title('AUC curves')
plt.plot(fpr0, tpr0, linestyle = '--', label = 'No Skill')
plt.plot(fpr1, tpr1, marker = '.', label = 'With 0-entries')
plt.plot(fpr2, tpr2, marker = '.', label = 'Without 0-entries')
plt.plot(fpr3, tpr3, marker = '.', label = 'Median Imputation')
plt.plot(fpr4, tpr4, marker = '.', label = 'KNN Imputation')

# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

# show the legend
plt.legend()

# show the plot
plt.show()