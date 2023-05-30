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

'''def one_hot_encode(df):
    # Get a list of columns to encode
    columns_to_encode = df.select_dtypes(include=['object']).columns.tolist()
    
    # Perform one-hot encoding
    encoded_df = pd.get_dummies(df, columns=columns_to_encode)
    print('encoded df: \n', encoded_df)
    
    return encoded_df'''


def divide_dataset(data_set):

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
    '''=========================I.D. DATA PREPARATION: FEATURE ENGINERING========================='''
    '''=========================I.E. DATA PREPARATION: TRAIN/TEST SPLIT========================='''
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
"""
    # Check which columns have NaN values after filling
    #print("Columns with NaN values after filling:\n", x.isna().sum())

    # Splitting the dataset into training and test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.20, random_state=40)

    '''=========================TRAINING DATA========================='''
    # Fitting Random Forest classifier to the training set
    classifier = RandomForestClassifier(n_estimators=400, max_depth=10, criterion="entropy")
    classifier.fit(x_train, y_train)

    '''=========================TESTING DATA========================='''
    # Predicting the test set result
    y_pred = classifier.predict(x_test)
    # Creating the confusion matrix and computing precision, recall, and F-beta score
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='bot')
    recall = recall_score(y_test, y_pred, pos_label='bot')
    f1beta = fbeta_score(y_test, y_pred, beta=2, pos_label='bot')
    # Printing the results and feature importance
    print("Confusion matrix: \n", cm)
    print("Precision: ", precision, "\nRecall: ", recall, "\nF-BETA: ", f1beta)
    print("Feature importance: ")
    for index, feature in enumerate(['followers_count',
              	'friends_count',
               	'listed_count',	
                'favourites_count',	
                'statuses_count',	
                'verified',	
                'description_text_mining_result',	
                'description_link_count',	
                'description_hashtags_count',	
                'description_mentions_count',	
                'tweet_rate']):
        print(feature,": ",classifier.feature_importances_[index])
    
    '''=========================HYPERPARAMETER TUNING========================='''
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

    '''=========================EXPORTING MODEL========================='''
    # Exporting the trained tree
    joblib.dump(classifier, 'model/classifier_accInfo.pkl')
    RandomForestTextMining = joblib.load('model/classifier_accInfo.pkl')
    # Get the list of classes
    classes = RandomForestTextMining.classes_
    # Print the classes
    print("Classes: ",classes)"""