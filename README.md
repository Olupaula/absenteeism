# A Modelling of Employee Absenteeism using Supervised Classifiers
 [Absenteeism](https://en.wikipedia.org/wiki/Absenteeism)  is a major problem when dealing with employees especially with 
 respect to assigning tasks, timeliness and achieving team goals at large. If a worker is frequently absent, the work flow sooner or later be 
 negatively affected, 
 unless the worker is replaced. This ordeal may be prevented in the recruitment process by developing an
 effective machine learning
 [model](https://learn.microsoft.com/en-us/windows/ai/windows-ml/what-is-a-machine-learning-model) to 
 predict whether an employee will be given to absenteeism.
 The absenteeism model is a 
 [classification model](https://learn.microsoft.com/en-us/training/modules/train-evaluate-classification-models/) 
 aimed at predicting whether a potential employee will be given to absenteeism or not.

 **Data Source**: The data was obtained from Machine Learning Repository

 **Classification Technique used**:

 1. [Logistic Regression (LOR)](https://online.stat.psu.edu/stat462/node/207/)
 2. [Support Vector Machine (SVM)](https://online.stat.psu.edu/stat857/node/211/)
 3. [K-Nearest Neighbor (KNN)](https://online.stat.psu.edu/stat508/lesson/k/)
 4. [Decision Tree (DT)](https://online.stat.psu.edu/stat857/node/236/)
 5. [Multi-Layer Perceptron Neural Network](https://www.researchgate.net/publication/354056558_Battle_royale_optimizer_for_training_multi-layer_perceptron)

 **Packages Used**: The [Scikit-learn (sklearn)](https://scikit-learn.org/) package was used to build these models. 
 The GridSearchCV of the 
 sklearn was used to tune the parameters of these models. For each of the models, an average validation score 
 and a test score were obtained after the models were trained. The best model was the one with 
 the highest validation score. The score is the accuracy of the model.
 Other tools used are [Pandas](https://scikit-learn.org/) and [Numpy](https://numpy.org/)

 **Evaluation Metric:** The 
 [accuracy score](https://developers.google.com/machine-learning/crash-course/classification/accuracy) 
 was used to evaluate the models. 

 **The best Model**:The higher the accuracy score, the better the model. The Decision Tree model 
 _(DecisionTreeClassifier(max_depth=7, max_features=3, max_leaf_nodes=14,random_state=123)_
 was found to be the best model, since it had the best accuracy score.


[View code on Kaggle](https://www.kaggle.com/code/oluade111/absenteeism-notebook/) 




