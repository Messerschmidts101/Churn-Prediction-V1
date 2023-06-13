"""
[4250 rows x 18 columns]
Mean Absolute Error (MAE):  32.47272527683262 # the closer to 0 the better
Mean Squared Error (MSE):  1662.2744489540348 # the closer to 0 the better
R-squared (R^2) score:  -0.018262025572180374 # the closer to 1 the better
state :  0.07485158202615687
area_code :  0.01680375290382754
international_plan :  0.006685973775438748
voice_mail_plan :  0.003197473075518161
number_vmail_messages :  0.037149315870202385
total_day_minutes :  0.06278094691487175
total_day_calls :  0.09947303201762905
total_day_charge :  0.06434617707201155
total_eve_minutes :  0.060961485290646066
total_eve_calls :  0.09996117300113577
total_eve_charge :  0.060022741287644774
total_night_minutes :  0.06797247171730819
total_night_calls :  0.09949527836281495
total_night_charge :  0.0590453556295583
total_intl_minutes :  0.05299626732839886
total_intl_calls :  0.04330414477448425
total_intl_charge :  0.0540876357464406
number_customer_service_calls :  0.03686519320591221

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
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
    # Extract rows of data that are churned only
    churn_data = data_set[data_set['churn'] == 'yes']
    churn_data.to_csv('model/data/testingchurneddata.csv', index=False)
    print("done!")
    # Extracting independent and dependent variable
    #x = data_set[[c for c in data_set.columns if c != 'churn' and data_set[c].dtype in ['float64', 'int64']]]
    x = data_set[[c for c in data_set.columns if c not in ['churn','account_length']]]
    y = data_set['account_length']
    print(x)
    '''=========================I.B. DATA PREPARATION: DATA IMPUTATION========================='''
    '''=========================I.C. DATA PREPARATION: DATA ENCODING========================='''
    x = one_hot_encode(x)
    print("after encoding: \n",x)
    '''=========================I.D. DATA PREPARATION: HANDLING IMBALANCED DATASET========================='''
    '''=========================I.E. DATA PREPARATION: FEATURE ENGINEERING========================='''
    '''=========================I.F. DATA PREPARATION: TRAIN/TEST SPLIT========================='''
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.20, random_state=123)
    '''=========================II. TRAINING========================='''
    classifier = RandomForestRegressor(n_estimators=400, max_depth=10, criterion="squared_error")
    classifier.fit(x_train, y_train)
    '''=========================III. TESTING AND EVALUATION========================='''
    # Predicting the test set result
    y_pred = classifier.predict(x_test)
    # Regression evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Printing the results and feature importance
    print("Mean Absolute Error (MAE): ", mae)
    print("Mean Squared Error (MSE): ", mse)
    print("R-squared (R^2) score: ", r2)
    for index, feature in enumerate([c for c in data_set.columns if c not in ['churn','account_length']]):
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
    classifier = RandomForestRegressor(n_estimators=grid_search.best_params_['n_estimators'], max_depth=grid_search.best_params_['max_depth'], criterion="squared_error")
    classifier.fit(x_train, y_train)
    print("classifier: ",classifier)
    '''=========================V.I EXPORTING: FINAL TESTING AND EVALUATION========================='''
    # Predicting the test set result
    y_pred = classifier.predict(x_test)
    # Regression evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Printing the results and feature importance
    print("Mean Absolute Error (MAE): ", mae)
    print("Mean Squared Error (MSE): ", mse)
    print("R-squared (R^2) score: ", r2)
    for index, feature in enumerate([c for c in data_set.columns if c not in ['churn','account_length']]):
        print(feature, ": ", classifier.feature_importances_[index])
    
    # Exporting the tree with the best hyperparameter
    joblib.dump(classifier, 'model/churnregressionprediction.pkl')
    
fitChurn()