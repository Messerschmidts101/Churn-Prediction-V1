"""
[4250 rows x 19 columns]
Confusion matrix: 
 [[738   4]
 [ 31  77]]
Precision:  0.9506172839506173
Recall:  0.7129629629629629
F-BETA:  0.7504873294346978
Feature importance:
state :  0.02091085773356425
account_length :  0.02364933536951014
area_code :  0.005446811097901077
international_plan :  0.09008765128451562
voice_mail_plan :  0.018082811468269604
number_vmail_messages :  0.02562426090857211
total_day_minutes :  0.15240994265109628
total_day_calls :  0.024252578124440406
total_day_charge :  0.1463627193163896
total_eve_minutes :  0.06244392907894073
total_eve_calls :  0.020166899254556468
total_eve_charge :  0.06205752452374809
total_night_minutes :  0.03484614057463236
total_night_calls :  0.02328064725325476
total_night_charge :  0.03403347558864758
total_intl_minutes :  0.03987570130753387
total_intl_calls :  0.04433202680088285
total_intl_charge :  0.036752017833653165
number_customer_service_calls :  0.13538466982989109

[[THINGS NEEDED TO WORK ON]]
1. Imputing missing values
2. Encoded variables [done]
3. Data Imputation
4. Handling imbalanced dataset
5. Feature engineering

"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, fbeta_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


def one_hot_encode(data_set):
    # Perform one-hot encoding
    for column in data_set.columns:
        if data_set[column].dtype not in ['float64', 'int64']:
            distinct_values = data_set[column].unique()  # Get distinct values
            mapping = {value: index for index, value in enumerate(distinct_values)}  # Create a mapping of values to indices
            print("Print mapping for column: ", column, " ||| ", mapping )
            for index, row in data_set.iterrows():
                value = row[column]
                if value in mapping:
                    data_set.at[index, column] = mapping[value]  # Replace value with corresponding index
    return data_set

def fitChurn():
    print("started")
    '''=========================I. DATA PREPARATION========================='''
    '''=========================I.A. DATA PREPARATION: DATA COLLECTION========================='''
    data_set = pd.read_csv('model/data/dataset.csv')
    # Extracting independent and dependent variable
    #x = data_set[[c for c in data_set.columns if c != 'churn' and data_set[c].dtype in ['float64', 'int64']]]
    x = data_set[[c for c in data_set.columns if c != 'churn']]
    y = data_set['churn']
    print(x)
    '''=========================I.B. DATA PREPARATION: DATA IMPUTATION========================='''
    '''=========================I.C. DATA PREPARATION: DATA ENCODING========================='''
    x = one_hot_encode(x)
    print("after encoding: \n",x)
    # Convert x to CSV
    x.to_csv('model/data/x_data.csv', index=False)
    '''=========================I.D. DATA PREPARATION: HANDLING IMBALANCED DATASET========================='''
    '''=========================I.E. DATA PREPARATION: FEATURE ENGINEERING========================='''
    '''=========================I.F. DATA PREPARATION: TRAIN/TEST SPLIT========================='''
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.20, random_state=123)
    '''=========================II. TRAINING========================='''
    classifier = RandomForestClassifier(n_estimators=400, max_depth=10, criterion="entropy")
    classifier.fit(x_train, y_train)
    '''=========================III. TESTING AND EVALUATION========================='''
    # Predicting the test set result
    y_pred = classifier.predict(x_test)
    # Creating the confusion matrix and computing precision, recall, and F-beta score
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='yes')
    recall = recall_score(y_test, y_pred, pos_label='yes')
    f1beta = fbeta_score(y_test, y_pred, beta=2, pos_label='yes')
    # Printing the results and feature importance
    print("Confusion matrix: \n", cm)
    print("Precision: ", precision, "\nRecall: ", recall, "\nF-BETA: ", f1beta)
    print("Feature importance: ")
    for index, feature in enumerate([c for c in data_set.columns if c != 'churn']):
        print(feature, ": ", classifier.feature_importances_[index])
    '''=========================IV. HYPERPARAMTER TUNING========================='''
    param_grid = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [5, 10, 15, 20]
    }
    # Perform grid search with 5-fold cross-validation
    grid_search = GridSearchCV(estimator = classifier, param_grid = param_grid, cv=5, n_jobs = -1, verbose = 2)
    grid_search.fit(x_train, y_train)
    # Print the best hyperparameters and corresponding score
    print("Best hyperparameters:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)
    '''=========================V. EXPORTING========================='''
    '''=========================V.I EXPORTING: FINAL TRAINING========================='''
    classifier = RandomForestClassifier(n_estimators=grid_search.best_params_['n_estimators'], max_depth=grid_search.best_params_['max_depth'], criterion="entropy")
    classifier.fit(x_train, y_train)
    print("classifier: ",classifier)
    '''=========================V.I EXPORTING: FINAL TESTING AND EVALUATION========================='''
    # Predicting the test set result
    y_pred = classifier.predict(x_test)
    # Creating the confusion matrix and computing precision, recall, and F-beta score
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='yes')
    recall = recall_score(y_test, y_pred, pos_label='yes')
    f1beta = fbeta_score(y_test, y_pred, beta=2, pos_label='yes')
    # Printing the results and feature importance
    print("Confusion matrix: \n", cm)
    print("Precision: ", precision, "\nRecall: ", recall, "\nF-BETA: ", f1beta)
    print("Feature importance: ")
    for index, feature in enumerate([c for c in data_set.columns if c != 'churn']):
        print(feature, ": ", classifier.feature_importances_[index])
    
    # Exporting the tree with the best hyperparameter
    joblib.dump(classifier, 'model/churnprediction.pkl')
    churnprediction = joblib.load('model/churnprediction.pkl')
    # Get the list of classes
    classes = churnprediction.classes_
    # Print the classes
    print("Classes: ",classes)
fitChurn()