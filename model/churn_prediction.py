"""
====================================================== As of 2nd July 2023 ====================================================== 
Best hyperparameters: {'max_depth': 20, 'n_estimators': 300}
Best score: 0.9967474542109519
classifier:  RandomForestClassifier(criterion='entropy', max_depth=20, n_estimators=300)
Confusion matrix: 
 [[740   6]
 [  0 715]]
Precision:  0.9916782246879334
Recall:  1.0
F-BETA:  0.9983244903658196
Feature importance:
NEW_total_local_charge :  0.13754365485469452
number_customer_service_calls :  0.13595946044285231
international_plan :  0.08353648370401262
total_day_charge :  0.06291220997811324
NEW_total_minutes :  0.06150507116886839
total_day_minutes :  0.054116405053873225
NEW_ratio_calls :  0.03759616875944958
total_intl_minutes :  0.027879427902315252
total_intl_charge :  0.027403255965258798
NEW_ratio_minutes :  0.023869134770129634
total_eve_minutes :  0.023609914739228934
NEW_ratio_charge :  0.023437135675494884
total_eve_charge :  0.023285382710469527
number_vmail_messages :  0.023156048903725142
NEW_avg_day_minutes :  0.02315083437010761
total_intl_calls :  0.02217516775411531
state :  0.019924526163628848
total_night_charge :  0.019651381993846508
total_night_minutes :  0.019533963122915998
voice_mail_plan :  0.019209753564183175
NEW_total_calls :  0.018682905356665124
total_night_calls :  0.018368475134569396
NEW_avg_night_minutes :  0.01821257703477922
total_eve_calls :  0.018010429245146047
total_day_calls :  0.01800511365023948
NEW_avg_eve_minutes :  0.017521083003001613
account_length :  0.016979113032361732
area_code :  0.004764921945953931
Classes:  ['no' 'yes']


====================================================== As of 15th June 2023 ====================================================== 
Best hyperparameters: {'max_depth': 20, 'n_estimators': 400}
Best score: 0.9957203499068401
classifier:  RandomForestClassifier(criterion='entropy', max_depth=20, n_estimators=400)
Confusion matrix: 
 [[741   5]
 [  2 713]]
Precision:  0.9930362116991643
Recall:  0.9972027972027973
F-BETA:  0.9963666852990497
Feature importance:
total_local_charge :  0.13664244645999618
number_customer_service_calls :  0.13362332851197506
international_plan :  0.08493836066204685
total_day_charge :  0.06544790925334067
total_minutes :  0.06417529140360671
total_day_minutes :  0.06297302722156543
ratio_calls :  0.03347293763206801
total_intl_minutes :  0.027246395345533508
total_intl_charge :  0.025560287874693274
ratio_charge :  0.025258110762905447
number_vmail_messages :  0.02504395138183666
ratio_minutes :  0.024261826311373115
total_intl_calls :  0.02314232866931457
total_eve_charge :  0.02312890598071331
total_eve_minutes :  0.02214924991000802
avg_day_minutes :  0.021412780555232068
total_night_charge :  0.019986879428898593
total_night_minutes :  0.019456798258613146
voice_mail_plan :  0.0192379081829739
total_eve_calls :  0.0180033842406327
total_calls :  0.017956802498495582
avg_night_minutes :  0.017796267638201958
state :  0.01772313888332333
total_night_calls :  0.016865893961226248
avg_eve_minutes :  0.01667829734849175
total_day_calls :  0.01659812045015082
account_length :  0.01648154063174197
area_code :  0.004737830541040989
Classes:  ['no' 'yes']

Print mapping for column:  state  |||  {'OH': 0, 'NJ': 1, 'OK': 2, 'MA': 3, 'MO': 4, 'LA': 5, 'WV': 6, 'IN': 7, 'RI': 8, 'IA': 9, 'MT': 10, 'NY': 11, 'ID': 12, 'VA': 13, 'TX': 14, 'FL': 15, 'CO': 16, 'AZ': 17, 'SC': 18, 'WY': 19, 'HI': 20, 'NH': 21, 'AK': 22, 'GA': 23, 'MD': 24, 'AR': 25, 'WI': 26, 'OR': 27, 'MI': 28, 'DE': 29, 'UT': 30, 'CA': 31, 
'SD': 32, 'NC': 33, 'WA': 34, 'MN': 35, 'NM': 36, 'NV': 37, 'DC': 38, 'VT': 39, 'KY': 40, 'ME': 41, 'MS': 42, 'AL': 43, 'NE': 44, 'KS': 45, 'TN': 46, 'IL': 47, 'PA': 48, 'CT': 49, 'ND': 50}
Print mapping for column:  area_code  |||  {'area_code_415': 0, 'area_code_408': 1, 'area_code_510': 2}
Print mapping for column:  international_plan  |||  {'no': 0, 'yes': 1}
Print mapping for column:  voice_mail_plan  |||  {'yes': 0, 'no': 1}


====================================================== As of previous version ====================================================== 
[4250 rows x 19 columns]
[('no', 3652), ('yes', 3652)] (7304,)
Confusion matrix: 
 [[734  12]
 [ 64 651]]
Precision:  0.9819004524886877
Recall:  0.9104895104895104
F-BETA:  0.9239284700539312
Feature importance:
state :  0.02237761832687284
account_length :  0.022072404490062873
area_code :  0.004788540574617004
international_plan :  0.10601417645318566
voice_mail_plan :  0.022119522133125652
number_vmail_messages :  0.027909219836398814
total_day_minutes :  0.1383517499660237
total_day_calls :  0.021452768204276534
total_day_charge :  0.15411747835220052
total_eve_minutes :  0.051406087741061486
total_eve_calls :  0.02122865528212054
total_eve_charge :  0.05541554086852904
total_night_minutes :  0.03254083362131864
total_night_calls :  0.020754283227885896
total_night_charge :  0.03319624585355292
total_intl_minutes :  0.03585100715225958
total_intl_calls :  0.0435754471003127
total_intl_charge :  0.036817624228211124
number_customer_service_calls :  0.15001079658798458
Classes:  ['no' 'yes']
"""

import re
from collections import Counter

import joblib
import nltk
import numpy as np
import pandas as pd
from imblearn import over_sampling, under_sampling
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (classification_report, confusion_matrix,
                             fbeta_score, precision_score, recall_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def label_encode(data_set,arrExcludedColumns):
    # Perform one-hot encoding
    for column in data_set.columns[~data_set.columns.isin(arrExcludedColumns)]:
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
    '''=========================I.B. DATA PREPARATION: DATA IMPUTATION========================='''
    for column in data_set.columns:
        if data_set[column].dtype == object:
            column_mode = data_set[column].mode().values[0]
            data_set[column].fillna(column_mode, inplace=True)
        else:
            column_mean = data_set[column].mean()
            data_set[column].fillna(column_mean, inplace=True)
    '''=========================I.C. DATA PREPARATION: DATA ENCODING========================='''
    data_set = label_encode(data_set,['churn'])
    print("after encoding: \n",data_set)

    '''=========================I.D. DATA PREPARATION: FEATURE ENGINEERING========================='''
    # Calculate the total number of calls made by a customer
    data_set['NEW_total_calls'] = data_set['total_day_calls'] + data_set['total_eve_calls'] + data_set['total_night_calls']
    # Comment: The total number of calls reflects the customer's overall engagement and usage of telecom services. Higher engagement may indicate a lower likelihood of churn.

    # Calculate the total number of minutes used by a customer
    data_set['NEW_total_minutes'] = data_set['total_day_minutes'] + data_set['total_eve_minutes'] + data_set['total_night_minutes']
    # Comment: Total minutes used represents the customer's level of activity and dependence on telecom services. Higher usage may suggest a stronger commitment to the service.

    # Calculate the ratio of international calls to total calls
    data_set['NEW_ratio_calls'] = data_set['total_intl_calls'] / data_set['NEW_total_calls']
    # Comment: The ratio of international calls helps identify the customer's international communication patterns. Higher ratios may indicate a specific need for international connectivity.

    # Calculate the ratio of international minutes to total minutes
    data_set['NEW_ratio_minutes'] = data_set['total_intl_minutes'] / data_set['NEW_total_minutes']
    # Comment: The ratio of international minutes highlights the extent of international calling relative to the overall calling activity. Higher ratios may indicate an increased risk of churn for customers with high international usage.

    # Calculate the average number of minutes used during day calls per month
    data_set['NEW_avg_day_minutes'] = data_set['total_day_minutes'] / data_set['account_length']
    # Comment: The average number of minutes used during day calls per month provides insights into the customer's typical daytime calling behavior. Unusually low or high values may be indicative of churn risk.

    # Calculate the average number of minutes used during evening calls per month
    data_set['NEW_avg_eve_minutes'] = data_set['total_eve_minutes'] / data_set['account_length']
    # Comment: The average number of minutes used during evening calls per month offers insights into the customer's evening calling habits and preferences. Deviations from the norm may signal churn risk.

    # Calculate the average number of minutes used during night calls per month
    data_set['NEW_avg_night_minutes'] = data_set['total_night_minutes'] / data_set['account_length']
    # Comment: The average number of minutes used during night calls per month provides insights into the customer's nighttime calling behavior and usage tendencies. Unusual patterns may indicate churn risk.

    # Calculate the total local charge incurred by a customer
    data_set['NEW_total_local_charge'] = data_set['total_day_charge'] + data_set['total_eve_charge'] + data_set['total_night_charge']
    # Comment: The total local charge reflects the customer's monetary commitment to local calls. Higher charges may indicate a stronger connection and lower churn likelihood.

    # Calculate the ratio of international charge to total local charge
    data_set['NEW_ratio_charge'] = data_set['total_intl_charge'] / data_set['NEW_total_local_charge']
    # Comment: The ratio of international charge to total local charge provides insights into the proportion of charges attributable to international calls. Higher ratios may suggest higher churn risk for customers with significant international charges.

    '''=========================I.E. DATA PREPARATION: HANDLING IMBALANCED DATASET========================='''
    # Extracting independent and dependent variable
    x = data_set[[c for c in data_set.columns if c != 'churn']]
    y = data_set['churn']
    print(x)
    x.to_csv('model/data/x_dataset.csv', index=False)
    rus = RandomOverSampler(sampling_strategy = 'minority')
    x_resampled, y_resampled = rus.fit_resample(x,y)
    print(sorted(Counter(y_resampled).items()),y_resampled.shape)
    '''=========================I.F. DATA PREPARATION: TRAIN/TEST SPLIT========================='''
    x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=.20, random_state=123)
    '''=========================II. TRAINING========================='''
    classifier = RandomForestClassifier(n_estimators=400, max_depth=20, criterion="entropy")
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
    
    # Create a dictionary to store the feature-importance pairs
    feature_importances = {}

    # Populate the dictionary with feature-importance pairs
    for index, feature in enumerate([c for c in data_set.columns if c != 'churn']):
        feature_importances[feature] = classifier.feature_importances_[index]

    # Sort the feature-importance dictionary in descending order based on the importance scores
    sorted_feature_importances = dict(sorted(feature_importances.items(), key=lambda x: x[1], reverse=True))

    # Print the feature names and their importance scores in descending order
    for feature, importance in sorted_feature_importances.items():
        print(feature, ": ", importance)
    
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
        # Create a dictionary to store the feature-importance pairs
    feature_importances = {}

    # Populate the dictionary with feature-importance pairs
    for index, feature in enumerate([c for c in data_set.columns if c != 'churn']):
        feature_importances[feature] = classifier.feature_importances_[index]

    # Sort the feature-importance dictionary in descending order based on the importance scores
    sorted_feature_importances = dict(sorted(feature_importances.items(), key=lambda x: x[1], reverse=True))

    # Print the feature names and their importance scores in descending order
    for feature, importance in sorted_feature_importances.items():
        print(feature, ": ", importance)
    
    # Exporting the tree with the best hyperparameter
    joblib.dump(classifier, 'model/churnprediction.pkl')
    churnprediction = joblib.load('model/churnprediction.pkl')
    # Get the list of classes
    classes = churnprediction.classes_
    # Print the classes
    print("Classes: ",classes)
fitChurn()