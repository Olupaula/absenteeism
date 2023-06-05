import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import joblib


# LOADING THE DATA
data = pd.read_csv("data/Absenteeism_at_work.csv", sep=";")
data2 = data

print(data.head())
print("The data is of dimension = ",  data.shape)


# EXPLORATORY DATA ANALYSIS
'''Data Cleaning'''
data_columns = list(data.columns)
print(data_columns)
data_columns[9] = "work_load_average_per_day"
data.columns = data_columns
columns = data.columns

# Dropping the ID column
data = data.drop(labels=['ID'], axis=1)

# Indices of categorical features
categorical_indices = [0, 1, 2, 3, 10, 11, 13, 14]
categorical_columns = data.iloc[:, categorical_indices].columns
print()
print('categorical indices =', categorical_indices)
print('categorical columns =', categorical_columns)

# Indices of numerical features
numerical_indices = [num for num in range(len(data.columns)) if num not in categorical_indices]
numerical_columns = data.iloc[:, numerical_indices].columns
print()
print('numerical indices =', numerical_indices)
print('numerical columns = ', numerical_columns)

# Is there any missing data?
any_na = data.isna().any()
print('\nAny missing data:')
print(any_na)

# Takes care of missing data for numerical features
for feature in numerical_columns:
    if data[feature].isna().any():
        data[feature] = data[feature].fillna(data[feature].mean())

# Takes care of missing data for categorical features
for feature in categorical_columns:
    if data[feature].isna().any():
        data[feature] = data[feature].fillna(data[feature].mode())

column_types = data.dtypes
print('\n Data types of the features:')
print(column_types)

# For proper visualization, the labels are clearly spelt out.
reason_for_absence = list(range(0, 28))
month_of_absence = list(range(0, 12))
day_of_the_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
day_of_the_week_code = [2, 3, 4, 5, 6]

seasons = ["Summer", "Autumn", "Winter", 'Spring']
seasons_code = [1, 2, 3, 4]

disciplinary_failure = ["Yes", "No"]
disciplinary_failure_code = [0, 1]

education = ["High School", "Graduate", "Postgraduate", "Master and Doctor"]
education_code = [1, 2, 3, 4]

social_drinker = ["Yes", "No"]
social_drinker_code = disciplinary_failure_code

social_smoker = ["Yes", "No"]
social_smoker_code = social_drinker_code

categories = [reason_for_absence,
              month_of_absence,
              day_of_the_week,
              seasons,
              disciplinary_failure,
              education,
              social_drinker,
              social_smoker]

categories_codes = [reason_for_absence,
                    month_of_absence,
                    day_of_the_week_code,
                    seasons_code,
                    disciplinary_failure_code,
                    education_code,
                    social_drinker_code,
                    social_smoker_code]
'''end of data cleaning '''

# Data visualization for categorical variables
k = 0
data_ = data.copy(deep=True)  # a copy of data
for i in categorical_indices:
    for j in range(0, len(categories[k])):
        data_.iloc[:, i] = data_.iloc[:, i].apply(lambda t: categories[k][j] if t == categories_codes[k][j] else t)
    k += 1

print(categories[3])

columns = data.columns

plot = {}
for i in categorical_indices:
    title = ("Plot of "
             + columns[i].title()
             + " against Absenteeism Time in Hours").replace("Of", "of").replace("For", "for").replace("The", "the")

    plt.title(title)
    plot[i] = sns.stripplot(x=data.iloc[:, i], y=data.iloc[:, -1], hue=data.iloc[:, i])
    if i == 4:
        sns.move_legend(plot[i], "lower left", bbox_to_anchor=(.4, .45))
    plt.show()


# From the plots for the following can be deduced:
# Infectious diseases seem to be the least reason for being absent, followed by perinatal period
# The month of absence has 13 months and so this column should be discarded
# The least absentee time in hours is on Thursdays
# # There is not much difference in absenteeism across the seasons winder
# More people seem to have absenteeism time in hours greater than 24 hours during Winter
# Those who don't have a record of disciplinary failure don't come late
# People with Masters or Doctorate degree have the least number of absenteeism time in hours.
# Social drinkers seem to be slightly more present than those who do not drink
# Those who do not smoke, generally, have much less absenteeism hours than those who do.

# Data visualization for numerical variables
plot = {}
for i in numerical_indices[1:-1]:  # indexing from 1 to -1 in order to exclude the target variable
    title = ("Plot of "
             + columns[i].title()
             + " against Absenteeism Time in Hours").replace(
            "Of", "of").replace("For", "for").replace("The", "the").replace("_", " ")

    plt.title(title)
    plot[i] = sns.scatterplot(x=data.iloc[:, i], y=data.iloc[:, -1])
    plt.show()

# From the plots, there is not a clear relationship between the numerical variables and absenteeism hours. Hence,
# the correlation coefficients are considered.


# Correlation between numerical features and absenteeism time in hours
correlations = data.iloc[:, numerical_indices].corr().iloc[:-2, -1]
print('\nCorrelation between numerical features and absenteeism time in hours:')
print(correlations)

# Correlation among the numerical features
pd.set_option('display.max_columns', None)
features_correlations = data.iloc[:, numerical_indices[1:-1]].corr()
print('\nCorrelation among features:')
print(features_correlations)

# Descriptive statistic for absenteeism time in hours
print('_'*70)
print('Descriptive statistic for absenteeism time in hours')
print('_'*70)

skewness = round(data.iloc[:, -1].skew(), 4)
print('\nSkewness =', skewness)
# Since the skewness is greater than 1, the absenteeism time in hours is extremely positively skewed.


# Kurtosis of absenteeism time in hours
kurtosis = round(data.iloc[:, -1].kurtosis(), 4)
print('\nKurtosis =', kurtosis)
# Since the kurtosis is greater than 3, it implies that the distribution of the absenteeism time in hours is more
# peaked than a normal distribution.

print('\nDescription:')
print(data.iloc[:, -1].describe())
print('_'*70, '\n')

# "Absenteeism time in hours" is recoded to "Absenteeism": (0 means employee will never be late, 1 means employee will
# be late however little)
data["absenteeism"] = data.loc[:, "Absenteeism time in hours"].map(
    lambda d:
    1 if d > data.loc[:, "Absenteeism time in hours"].median()
    else 0
)
print('The new column called absenteeism:')
print(data.loc[:, "absenteeism"])


# 3) FEATURE SELECTION
y = data["absenteeism"]
x = data.drop(labels=["Absenteeism time in hours", "absenteeism"], axis=1)

# Selecting the features to be included based on Chi-square scores
feature_selector = SelectKBest(score_func=chi2, k=4)
fit = feature_selector.fit(x, y)
feature_data = fit.transform(x)
selected_features = feature_selector.get_feature_names_out(input_features=None)
print('\n', ''"selected_x = ", selected_features)
print("selection scores =", feature_selector.scores_.round(), '\n')

# A New dataframe for the selected features called x
x = data.loc[:, feature_selector.get_feature_names_out(input_features=None)]

# print(x)

# Splitting the data to train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=123)

# lists needed to display the accuracies of the models and save them
joblib_file_names = [
                "LogisticRegression.joblib",
                "SupportVectorMachine.joblib",
                "K_NearestNeighbour.joblib",
                "DecisionTree.joblib"
]  # A list of the filenames of the models to be serialized by joblib.

models = []  # A list of the model objects to be serialized by joblib
model_name = ["Logistic Regression", "Support Vector Machine", "K-Nearest Neighbour", "Decision Tree"]
average_validation_accuracy = []
test_accuracy = []
estimator = []  # the model with the recommended parameters after tuning


# 4) THE MODELS
n_ = np.array(range(1, 10))
cr = [10 ** n for n in n_]  # c's greater than or equal to one
cl = [10/(10 ** n) for n in n_]  # c's less than 1
C = cl + cr  # both c's
C.sort()


# (a) LOGISTIC REGRESSION MODEL (LOR)
"""
logistic_model = LogisticRegression(random_state=123)
parameters = {"C": C, "max_iter": [10000, 15000], "solver": ["saga", "newton-cg", "sag", "newton-cholesky"]}

# Tuning the logistic model to find the best model parameters which make up the best estimator
print("Searching for the best LOR model ...")
clf = GridSearchCV(logistic_model, parameters)
clf.fit(x_train, y_train)

# best estimator for the logistic regression
best_lor_estimator = clf.best_estimator_
"""

best_lor_estimator = LogisticRegression(C=100, max_iter=10000, random_state=123, solver='newton-cg')

# Fitting the selected (best) model
best_lor_estimator.fit(x_train, y_train)

# Evaluating the accuracy of the model
validation_scores = cross_val_score(best_lor_estimator, X=x_train, y=y_train,  cv=5)
lor_average_validation_accuracy = round(validation_scores.mean(), 4)
lor_test_accuracy = round(best_lor_estimator.score(x_test, y_test), 4)

# output for logistic regression
print("_"*100)
print("Output for Logistic Regression (LOR)")
print("_"*100)
print("The best LOR model:", best_lor_estimator)
print("LOR Validation scores = ", validation_scores.round(decimals=4))
print("LOR Average validation scores = ", lor_average_validation_accuracy)
print("LOR Test Score = ", lor_test_accuracy)
print()

# adding the logistic regression model scores to the lists above
average_validation_accuracy.append(lor_average_validation_accuracy)
test_accuracy.append(lor_test_accuracy)
estimator.append(best_lor_estimator)


# (b) SUPPORT VECTOR MACHINE (SVM)
# The Support Vector Machine Model (SVM)
""" 
svc = SVC(random_state=123)
parameters = {"C": [0.01, 10], "gamma": range(0, 2), "kernel": ["linear", "rbf"]}

# Tuning the svm model to find the best model parameters which make up the best estimator
print("Searching for the best SVM model...")
clf = GridSearchCV(svc, parameters)
clf.fit(x_train, y_train)

# best estimator for the svm
best_svm_estimator = clf.best_estimator_
"""


best_svm_estimator = SVC(C=10, gamma=1, random_state=123)

# fitting the selected (best) model
best_svm_estimator.fit(x_train, y_train)

# Evaluating the accuracy of the model
svm_validation_scores = cross_val_score(best_svm_estimator, x_train, y_train, cv=5)
svm_average_validation_score = round(svm_validation_scores.mean(), 4)
svm_test_accuracy = round(best_svm_estimator.score(x_test, y_test), 4)

# output for svm
print("_"*100)
print("output for svm")
print("_"*100)
print("The best svm model:", best_svm_estimator)
print("SVM Validation scores = ", svm_validation_scores)
print("SVM Validation Score = ", svm_average_validation_score)
print("SVM Test_Accuracy", svm_test_accuracy)
print()

# adding the svm estimator and model scores to the lists above
average_validation_accuracy.append(svm_average_validation_score)
test_accuracy.append(svm_test_accuracy)
estimator.append(best_svm_estimator)


# (c) K-NEAREST NEIGHBOUR (KNN)

# The KNN model
"""
knn_model = KNeighborsClassifier()
parameters = {"n_neighbors": range(3, 21, 2)}

# Tuning the knn model to find the best model parameters which make up the best estimator
print("Searching for the best KNN model...")
clf = GridSearchCV(knn_model, parameters)
clf.fit(x_train, y_train)

# best estimator for the knn model
best_knn_estimator = clf.best_estimator_
"""

best_knn_estimator = KNeighborsClassifier(n_neighbors=3)

# Fitting the selected (best) model
best_knn_estimator.fit(x_train, y_train)

# Evaluating the accuracy of the model
validation_scores = cross_val_score(best_knn_estimator, X=x_train, y=y_train,  cv=5)
knn_average_validation_accuracy = round(validation_scores.mean(), 4)
knn_test_accuracy = round(best_knn_estimator.score(x_test, y_test), 4)

print("_"*100)
print("Output for K-Nearest Neighbours (KNN) Model")
print("_"*100)
print("The best KNN model:", best_knn_estimator)
print("KNN Validation scores = ", validation_scores.round(decimals=4))
print("KNN Average validation scores = ", knn_average_validation_accuracy)
print("KNN Test Score = ", knn_test_accuracy)
print()

# Adding the KNN estimator and model scores to the lists above
average_validation_accuracy.append(knn_average_validation_accuracy)
test_accuracy.append(knn_test_accuracy)
estimator.append(best_knn_estimator)


# (d) DECISION TREE (DT)
# The DT model
""" 
dt_model = DecisionTreeClassifier(random_state=123)
parameters = {"max_depth": range(2, 20),
              'max_features': range(2, 6),
              'max_leaf_nodes': range(2, 20)}
dt_model.fit(x_train, y_train)

# Tuning the DT model to find the best model parameter which make up the best estimator
 print("Searching for the best DT model...")
clf = GridSearchCV(dt_model, parameters)
clf.fit(x_train, y_train)

# best estimator for the knn model
best_knn_estimator = clf.best_estimator_
"""

# The best estimator for the logistic regression
# best_dt_estimator = clf.best_estimator_
best_dt_estimator = DecisionTreeClassifier(max_depth=7, max_features=3, max_leaf_nodes=14, random_state=123)

# Fitting the selected (best) model
best_dt_estimator.fit(x_train, y_train)

# Evaluating the accuracy of the model
validation_scores = cross_val_score(best_dt_estimator, X=x_train, y=y_train,  cv=5)
dt_average_validation_accuracy = round(validation_scores.mean(), 4)
dt_test_accuracy = round(best_dt_estimator.score(x_test, y_test), 4)

# Output for the DT regression
print("_"*100)
print("Output for Decision Tree (DT) Model")
print("_"*100)
print("The best DT Model = ", best_dt_estimator)
print("DT Validation scores = ", validation_scores.round(decimals=4))
print("DT Average validation scores = ", dt_average_validation_accuracy)
print("DT Test Score = ", dt_test_accuracy)
print()

# Adding the DT estimator and model scores to the lists above
average_validation_accuracy.append(dt_average_validation_accuracy)
test_accuracy.append(dt_test_accuracy)
estimator.append(best_dt_estimator)


# 5) SUMMARY OF THE PERFORMANCES OF THE MODELS
# The data frame for comparing the accuracy scores of the models
print("_"*100)
print("Summary of the Performances of the Models")
print("_"*100)
models_and_accuracies = pd.DataFrame({"Model Name": model_name, "estimator": estimator,
                                      "Average Validation Accuracy": average_validation_accuracy,
                                      "Test Accuracy": test_accuracy})

print(models_and_accuracies)
m_and_a = models_and_accuracies
for i in range(len(models_and_accuracies)):
    if m_and_a.iloc[i, 2] == m_and_a.iloc[:, 2].max() and m_and_a.iloc[i, 3] == m_and_a.iloc[:, 3].max():
        print("The best model with respect to Validation Accuracy and Test Accuracy:")
        print(m_and_a.loc[i])
    else:
        if m_and_a.iloc[i, 2] == m_and_a.iloc[:, 2].max():
            print("Best Model According to Validation Accuracy:")
            print(m_and_a.iloc[i, 2])


# 6) SAVING THE MODELS
for i in range(len(joblib_file_names)):
    joblib.dump(estimator[i], "models/" + joblib_file_names[i])

# 7) USING THE BEST MODEL FOR PREDICTION
# (whether workers will be given to absenteeism or not)
file = "models/DecisionTree.joblib"
dt_model = joblib.load(file)
y_predicted = dt_model.predict(x_test)
y_y_predicted = pd.DataFrame(np.array([y_test, y_predicted])).T
y_y_predicted.columns = ["Original y", "Predicted y"]

# Recoding the responses: A = 1, which means Given to Absenteeism and N = 1, which means Not Given to Absenteeism
y_y_predicted["Original y"] = y_y_predicted["Original y"].map(lambda fx: "A" if fx == 1 else "N")
y_y_predicted["Predicted y"] = y_y_predicted["Predicted y"].map(lambda fx: "A" if fx == 1 else "N")
y_y_predicted.to_csv("prediction/absenteeism_classification_result.csv")
