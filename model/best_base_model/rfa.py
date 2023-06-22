import base64
import csv
from itertools import product
import json
import math
import pickle
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_curve, fbeta_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight
from tqdm import tqdm
from dataset import load_dataset
from gen_feat import generate_features, print_corr
from main_forest import preprocess_numeric_features, transform

def load_feats(top_list=False, return_index=False):
    """
    Load the selected features from the feature ranking file.

    Parameters:
        top_list (bool): Whether to return the top feature lists.
        return_index (bool): Whether to return feature names or indices.

    Returns:
        selected_features (list): List of selected feature names or indices.
            If top_list is False, it contains all selected features.
            If top_list is True, it contains multiple lists, each with the top features.
    """
    feat = pd.read_csv('model/data/feature_ranking_support.csv')

    if return_index:
        selected_features = feat.loc[(feat['Ranking'] == 1)].index.tolist()
    else:
        selected_features = feat.loc[(feat['Ranking'] == 1), 'Feature'].tolist()
        print(selected_features)

    if top_list:
        top_features = [selected_features[:count] for count in [30,25,20, 15, 10, 5]]
        return top_features

    return selected_features

def evaluate_model(model, x_train, y_train, x_test, y_test):
    """
    Evaluate a given model on the test set and calculate the accuracy.

    Args:
        model (object): The model to evaluate.
        x_train (numpy.ndarray): The feature matrix of the training set.
        y_train (numpy.ndarray): The target vector of the training set.
        x_test (numpy.ndarray): The feature matrix of the test set.
        y_test (numpy.ndarray): The target vector of the test set.

    Returns:
        float: The accuracy score of the model on the test set.
        numpy.ndarray: The predicted labels for the test set.
    """
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, y_pred

def load_data():
    # Load your data here
    x = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test

def train_base_classifier(x_train, y_train, class_weights, base_classifier=None):
    if base_classifier is None:
        base_classifier = RandomForestClassifier(
            criterion='entropy', 
            max_depth=15, 
            max_features=None, 
            min_samples_leaf=1, 
            min_samples_split=2, 
            n_estimators=100,
            class_weight=class_weights,
            random_state=42
            )
    base_classifier.fit(x_train, y_train)
    return base_classifier

def save_to_json(text, filename='best_features'):
    # Path to the JSON file
    file_path = "data/" + filename + ".json"
    # Save the list as JSON
    with open(file_path, "w") as file:
        json.dump(text, file)

def get_class_weights(df):
    # Calculate class weights based on the imbalance ratio
    imbalance_ratio = np.array([[720, 2], [19, 110]])
    total_samples = np.sum(imbalance_ratio)
    class_weights = {
        0: total_samples / (2 * imbalance_ratio[0, 0]),
        1: total_samples / (2 * imbalance_ratio[1, 1])
    }

def save_selected_features_text(selected_features, file_path):
    with open(file_path, 'w') as file:
        for feature in selected_features:
            file.write(feature + '\n')

def save_as_csv(ranking, support, feature_names, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Feature', 'Ranking', 'Support'])
        for feature, rank, sup in zip(feature_names, ranking, support):
            writer.writerow([feature, rank, sup])

def rfa_bagging_rfecv(df, x_train, x_test, y_train, y_test, param_grid=None, base_classifier=None, perform = ['grid_search', 'rfecv'], cv=5, step=10, feat_done = False, results_file = None):
    
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    
    if param_grid is None:
        param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15],
        'criterion': ['gini', 'entropy'],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }

    if base_classifier is None:
        base_classifier= RandomForestClassifier(criterion='entropy', max_depth=15, max_features=None, min_samples_leaf=1, min_samples_split=2, n_estimators=100,class_weight=dict(enumerate(class_weights)),random_state=42)

        base_classifier.fit(x_train, y_train)
        y_pred = base_classifier.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print('Base Model Accuracy:', accuracy)

    if 'grid_search' in perform:
        # Train the base RandomForestClassifier with grid search
        grid_search = GridSearchCV(estimator=base_classifier, param_grid=param_grid, cv=cv, verbose=2)

        param_combinations = list(product(param_grid['n_estimators'], param_grid['max_depth']))
        total_iterations = len(param_combinations)*cv

        with tqdm(total=total_iterations, desc="Grid Search") as pbar:
            for estimator, depth in param_combinations:
                base_classifier.set_params(n_estimators=estimator, max_depth=depth)
                grid_search.estimator = base_classifier
                grid_search.fit(x_train, y_train)
                pbar.update(1)

            best_params = grid_search.best_estimator_.get_params()
            best_score = grid_search.best_score_
            pbar.set_postfix({'Best_Params': best_params, 'Best_Score': best_score, 'Mean_score': accuracy})
                
        best_base_classifier = grid_search.best_estimator_
        # Train the best base classifier
        best_base_classifier.fit(x_train, y_train)

        # Save the best estimator
        joblib.dump(best_base_classifier, 'model/data/best_base_estimator.pkl')
    else:
        best_base_classifier = base_classifier
    '''
    
    best_base_classifier = RandomForestClassifier(criterion='entropy', max_depth=15, max_features=None, min_samples_leaf=1, min_samples_split=2, n_estimators=100,class_weight=dict(enumerate(class_weights)),random_state=42)

    Grid Search: 486it [7:55:22, 58.69s/it, Best_Params={'criterion': 'entropy', 'max_depth': 15, 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}, Best_Score=0.979, Mean_score=0.978]
    '''
    df = df.drop('churn', axis=1)
    if 'churn' in x_train.columns:
        x_train = x_train.drop('churn', axis=1)
    valid = x_train.columns.equals(df.columns)
    print(x_train.shape[1], df.shape[1])
    print("VALIDATOR",valid)
    
    if valid and feat_done == False:
    # Perform RFECV to select the most relevant features
        with tqdm(total=1, desc="Feature Selection") as pbar:
            selector = RFECV(estimator=best_base_classifier, step=step, cv=StratifiedKFold(cv), scoring='accuracy', verbose=2)
            selector.fit(x_train, y_train)
            x_train_selected = selector.transform(x_train)
            selected_indices = selector.support_
            feature_names = df.columns.tolist()
            ranking = selector.ranking_
            support = selector.support_
            filename = 'model/data/feature_ranking_support.csv'
            save_as_csv(ranking, support, feature_names, filename)
            save_data_as_csv(x_train, y_train, x_test, y_test)
            feat_done = True
            pbar.update(1)
    models = {}  # Initialize as an empty dictionary
    models['Base_Model'] = best_base_classifier
    if  feat_done:
        ada_model = AdaBoostClassifier(base_estimator=best_base_classifier, n_estimators=400)
        ada_model.fit(x_train, y_train)
    else:
        ada_model.fit(x_train_selected, y_train)
        models['AdaBoost_Model']= ada_model
    # Create a list of models and their names
    feature_list = load_feats()
    print('MY FEATURE LIST:',feature_list)
    interval = create_interval(5,len(feature_list), 50)
    print(interval)
    results = evaluate_models(models, interval , x_train, y_train, x_test, y_test, params=param_grid, step = step, results_file= results_file, feature_list = feature_list)
    return results

def create_interval(start, length, max_value):
    step = math.ceil((max_value - start + 1) / (length))
    interval = [start + i * step for i in range(length)]
    interval = [x for x in interval if x <= max_value]
    return interval

def evaluate_model(model, x_train, y_train, x_test, y_test):
    """
    Evaluate a given model on the test set and calculate the accuracy.

    Args:
        model (object): The model to evaluate.
        x_train (numpy.ndarray): The feature matrix of the training set.
        y_train (numpy.ndarray): The target vector of the training set.
        x_test (numpy.ndarray): The feature matrix of the test set.
        y_test (numpy.ndarray): The target vector of the test set.

    Returns:
        float: The accuracy score of the model on the test set.
        numpy.ndarray: The predicted labels for the test set.
    """
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, y_pred


def evaluate_feature_lists_has_feat_input(model, param_grid, feature_lists, all_features, x_train, y_train, x_test, y_test):
    """
    Evaluate the model using different feature subsets and select the best subset.

    Args:
        model (object): The model to evaluate.
        param_grid (dict): The parameter grid for grid search (optional).
        feature_lists (list): List of feature subsets to evaluate.
        all_features (list): List of all feature names.
        x_train (numpy.ndarray): The feature matrix of the training set.
        y_train (numpy.ndarray): The target vector of the training set.
        x_test (numpy.ndarray): The feature matrix of the test set.
        y_test (numpy.ndarray): The target vector of the test set.

    Returns:
        list: List of accuracy scores for each feature subset.
        object: The best model selected based on the highest accuracy score (if param_grid is provided),
                otherwise None.
        list: The best features selected by the model as a list of feature names (if no param_grid is provided),
              otherwise None.
    """
    accuracies = []
    best_model = None
    best_accuracy = 0.0
    best_features = None

    for features in feature_lists:
        # Convert feature names to indices
        feature_indices = get_feature_indices(features, all_features)
        x_train_selected = x_train[:, feature_indices]
        x_test_selected = x_test[:, feature_indices]

        if param_grid is not None:
            # Perform grid search with 10-fold cross-validation
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10)
            grid_search.fit(x_train_selected, y_train)

            # Get the best model and predict on the test set
            y_pred = grid_search.predict(x_test_selected)

            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)

            # Save the best model if it has the highest accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = grid_search.best_estimator_
        else:
            # Fit the model on the training set and predict on the test set
            model.fit(x_train_selected, y_train)
            y_pred = model.predict(x_test_selected)

            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)

            # Save the best features and model if it has the highest accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_features = [all_features[index] for index in feature_indices]
                best_model = model

        accuracies.append(accuracy)

        # Print and save the metrics for the current feature subset
        print_metrics(y_test, y_pred)

    return accuracies, best_model, best_features

def evaluate_feature_lists_2(model, param_grid, feature_sizes, x_train, y_train, x_test, y_test, step=5):
    """
    Evaluate the model using different feature subsets and select the best subset based on feature importances.

    Args:
        model (object): The model to evaluate.
        param_grid (dict): The parameter grid for grid search (optional).
        feature_sizes (list): The desired feature sizes to evaluate.
        x_train (numpy.ndarray): The feature matrix of the training set.
        y_train (numpy.ndarray): The target vector of the training set.
        x_test (numpy.ndarray): The feature matrix of the test set.
        y_test (numpy.ndarray): The target vector of the test set.
        step (int): The number of features to remove at each iteration. Default is 5.

    Returns:
        list: List of accuracy scores for each feature subset.
        object: The best model selected based on the highest accuracy score (if param_grid is provided),
                otherwise None.
        list: The best features selected by the model as a list of feature names (if no param_grid is provided),
              otherwise None.
    """
    accuracies = []
    best_model = None
    best_accuracy = 0.0
    best_features = None

    num_features = x_train.shape[1]
    all_features = [f"Feature {i+1}" for i in range(num_features)]

    for feature_size in feature_sizes:
        while num_features > feature_size:
            # Train the model and get feature importances
            model.fit(x_train, y_train)
            feature_importances = model.feature_importances_

            # Sort feature importances in descending order
            sorted_indices = np.argsort(feature_importances)[::-1]

            if num_features - feature_size >= step:
                # Select top features based on step size
                selected_indices = sorted_indices[:num_features - step]
            else:
                # Select top features based on feature_size
                selected_indices = sorted_indices[:feature_size]

            selected_features = [all_features[index] for index in selected_indices]

            # Update feature matrix with selected features
            x_train_selected = x_train[:, selected_indices]
            x_test_selected = x_test[:, selected_indices]

            if param_grid is not None:
                # Perform grid search with 10-fold cross-validation
                grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10)
                grid_search.fit(x_train_selected, y_train)

                # Get the best model and predict on the test set
                y_pred = grid_search.predict(x_test_selected)

                # Calculate accuracy
                accuracy = accuracy_score(y_test, y_pred)

                # Save the best model if it has the highest accuracy
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = grid_search.best_estimator_
            else:
                # Fit the model on the training set and predict on the test set
                model.fit(x_train_selected, y_train)
                y_pred = model.predict(x_test_selected)

                # Calculate accuracy
                accuracy = accuracy_score(y_test, y_pred)

                # Save the best features and model if it has the highest accuracy
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_features = selected_features
                    best_model = model

            accuracies.append(accuracy)

            # Print and save the metrics for the current feature subset
            print(f"Evaluated Features (Feature Size: {num_features}): {selected_features}")
            print_metrics(y_test, y_pred)

            # Update all_features, x_train, and num_features for the next iteration
            all_features = selected_features
            x_train = x_train_selected
            num_features = x_train.shape[1]

    return accuracies, best_model, best_features

def get_feature_indices(feature_names, all_features):
    """
    Get the indices of the selected feature names from the list of all features.

    Args:
        feature_names (list): List of feature names to get indices for.
        all_features (list): List of all feature names.

    Returns:
        list: List of feature indices corresponding to the selected feature names.
    """
    feature_indices = []
    for feature_name in feature_names:
        try:
            index = all_features.index(feature_name)
            feature_indices.append(index)
        except ValueError:
            print(f"Feature '{feature_name}' not found in the list of all features.")

    return feature_indices

def evaluate_feature_lists_(model, param_grid, feature_sizes, x_train, y_train, x_test, y_test, feature_lists=None):
    """
    Evaluate the model using different feature subsets and select the best subset based on feature importances.

    Args:
        model (object): The model to evaluate.
        param_grid (dict): The parameter grid for grid search (optional).
        feature_sizes (list): The desired feature sizes to evaluate.
        x_train (numpy.ndarray): The feature matrix of the training set.
        y_train (numpy.ndarray): The target vector of the training set.
        x_test (numpy.ndarray): The feature matrix of the test set.
        y_test (numpy.ndarray): The target vector of the test set.
        feature_lists (list): List of feature lists to evaluate (optional).

    Returns:
        list: List of accuracy scores for each feature subset.
        object: The best model selected based on the highest accuracy score (if param_grid is provided),
                otherwise None.
        list: The best features selected by the model as a list of feature names (if no param_grid is provided),
              otherwise None.
    """
    accuracies = []
    best_model = None
    best_accuracy = 0.0
    best_features = None

    num_features = x_train.shape[1]
    all_features = [f"Feature {i + 1}" for i in range(num_features)]

    if feature_lists is None:
        feature_lists = [all_features]

    for feature_list in feature_lists:
        selected_indices = [i for i, feature in enumerate(all_features) if feature in feature_list]
        selected_features = [all_features[index] for index in selected_indices]

        # Update feature matrix with selected features
        x_train_selected = x_train.iloc[:, selected_indices]
        x_test_selected = x_test.iloc[:, selected_indices]

        # Fit the model on the training set and predict on the test set
        model.fit(x_train_selected, y_train)
        y_pred = model.predict(x_test_selected)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Save the best features and model if it has the highest accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_features = selected_features
            best_model = model

        if param_grid is not None:
            # Perform grid search with 10-fold cross-validation
            grid_search = GridSearchCV(estimator=best_model, param_grid=param_grid, cv=10)
            grid_search.fit(x_train_selected, y_train)

            # Get the best model and predict on the test set
            y_pred = grid_search.predict(x_test_selected)

            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)

            # Save the best model if it has the highest accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = grid_search.best_estimator_

        accuracies.append(accuracy)
        # Print and save the metrics for the current feature subset
        print(f"Evaluated Features (Feature List: {selected_features}): {selected_features}")
        print_metrics(y_test, y_pred)

    return accuracies, best_model, best_features

def evaluate_feature_lists_working(model, param_grid, feature_sizes, x_train, y_train, x_test, y_test, step=1):
    """
    Evaluate the model using different feature subsets and select the best subset based on feature importances.

    Args:
        model (object): The model to evaluate.
        param_grid (dict): The parameter grid for grid search (optional).
        feature_sizes (list): The desired feature sizes to evaluate.
        x_train (pandas.DataFrame): The feature matrix of the training set.
        y_train (numpy.ndarray): The target vector of the training set.
        x_test (pandas.DataFrame): The feature matrix of the test set.
        y_test (numpy.ndarray): The target vector of the test set.
        step (int): The number of features to remove in each iteration. Default is 1.

    Returns:
        list: List of accuracy scores for each feature subset.
        object: The best model selected based on the highest accuracy score (if param_grid is provided),
                otherwise None.
        list: The best features selected by the model as a list of feature names (if no param_grid is provided),
              otherwise None.
    """
    if step <= 0:
        raise ValueError("Step value must be greater than 0.")

    accuracies = []
    best_model = None
    best_accuracy = 0.0
    best_features = None

    num_features = x_train.shape[1]
    all_features = x_train.columns.tolist()

    for k in tqdm(feature_sizes, desc="Feature Sizes"):
        remaining_features = all_features.copy()

        p_bar = tqdm(total=len(remaining_features), desc="Feature Elimination", leave=False)

        while len(remaining_features) >= k:
            p_bar.update(step)

            # Update feature matrix with remaining features
            x_train_selected = x_train[remaining_features]
            x_test_selected = x_test[remaining_features]

            # Fit the model on the training set and predict on the test set
            model.fit(x_train_selected, y_train)
            y_pred = model.predict(x_test_selected)

            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)

            # Save the best features and model if it has the highest accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_features = remaining_features.copy()
                best_model = model

            accuracies.append(accuracy)

            if len(remaining_features) == k:
                break

            # Get feature importances and eliminate the least important features
            importances = model.feature_importances_
            least_important_feature_indices = np.argsort(importances)[:step]
            least_important_features = [remaining_features[i] for i in least_important_feature_indices]

            for feature in least_important_features:
                remaining_features.remove(feature)

            # Update the progress bar
            p_bar.total = len(remaining_features)
            p_bar.refresh()

        p_bar.close()

    if param_grid is not None:
        # Update feature matrix with best features
        x_train_selected = x_train[best_features]
        x_test_selected = x_test[best_features]

        # Perform grid search with 10-fold cross-validation
        grid_search = GridSearchCV(estimator=best_model, param_grid=param_grid, cv=10)
        grid_search.fit(x_train_selected, y_train)

        # Get the best model and predict on the test set
        y_pred = grid_search.predict(x_test_selected)

        # Calculate accuracy
        best_accuracy = accuracy_score(y_test, y_pred)
        best_model = grid_search.best_estimator_

    return accuracies, best_model, best_features

from itertools import product

def evaluate_feature_lists_latest(model, param_grid, feature_sizes, x_train, y_train, x_test, y_test, step=1, gridsearch=True):
    """
    Evaluate the model using different feature subsets and select the best subset based on feature importances.

    Args:
        model (object): The model to evaluate.
        param_grid (dict): The parameter grid for grid search (optional).
        feature_sizes (list): The desired feature sizes to evaluate.
        x_train (pandas.DataFrame): The feature matrix of the training set.
        y_train (pandas.Series or numpy.ndarray): The target vector of the training set.
        x_test (pandas.DataFrame): The feature matrix of the test set.
        y_test (pandas.Series or numpy.ndarray): The target vector of the test set.
        step (int): The number of features to remove in each iteration. Default is 1.
        gridsearch (bool): Whether to perform grid search. Default is True.

    Returns:
        float: Mean performance of grid search (if gridsearch is True and param_grid is provided),
               otherwise None.
        list: List of accuracy scores for each feature subset.
        object: The best model selected based on the highest accuracy score (if gridsearch is True),
                otherwise None.
        list: The best features selected by the model as a list of feature names (if gridsearch is False),
              otherwise None.
    """
    if step <= 0:
        raise ValueError("Step value must be greater than 0.")

    accuracies = []
    best_model = None
    best_accuracy = 0.0
    best_features = None

    num_features = x_train.shape[1]
    all_features = x_train.columns.tolist()

    for k in tqdm(feature_sizes, desc="Feature Sizes"):
        remaining_features = all_features.copy()

        p_bar = tqdm(total=len(remaining_features), desc="Feature Elimination", leave=False, postfix="")

        while len(remaining_features) >= k:
            p_bar.update(step)

            # Update feature matrix with remaining features
            x_train_selected = x_train[remaining_features]
            x_test_selected = x_test[remaining_features]

            # Fit the model on the training set and predict on the test set
            model.fit(x_train_selected, y_train)
            y_pred = model.predict(x_test_selected)

            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)

            # Save the best features and model if it has the highest accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_features = remaining_features.copy()
                best_model = model

            accuracies.append(accuracy)

            if len(remaining_features) == k:
                break

            # Get feature importances and eliminate the least important features
            importances = model.feature_importances_
            least_important_feature_indices = np.argsort(importances)[:step]
            least_important_features = [remaining_features[i] for i in least_important_feature_indices]

            for feature in least_important_features:
                remaining_features.remove(feature)
                best_features = remaining_features.copy()

            # Update the progress bar postfix with accuracies for each iteration
            pbar_postfix = f"Accuracies: {accuracies[-k:]}, Mean: {np.mean(accuracies[-k:])}, Features: {remaining_features}"
            p_bar.set_postfix_str(pbar_postfix)
            p_bar.total = len(remaining_features)
            p_bar.refresh()

        p_bar.close()

    mean_accuracy = np.mean(accuracies)

    if gridsearch:
        print("Performing Grid Search for Model:", model.__class__.__name__)
        print("Features:", best_features)
        print("Accuracies:", accuracies)
        print('Mean Accurary:', mean_accuracy)

        # Perform grid search with 10-fold cross-validation
        grid_results = []
        param_combinations = list(product(*param_grid.values()))
        total_combinations = len(param_combinations)
        pbar_grid = tqdm(total=total_combinations, desc="Grid Search")

        for params in param_combinations:
            pbar_grid.set_postfix_str(f"Params: {params}")
            best_model.set_params(**dict(zip(param_grid.keys(), params)))

            # Update feature matrix with best features
            x_train_selected = x_train[best_features]
            x_test_selected = x_test[best_features]

            # Fit the model on the training set and predict on the test set
            model.fit(x_train_selected, y_train)
            y_pred = model.predict(x_test_selected)

            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            grid_results.append(accuracy)

            pbar_grid.update(1)

        pbar_grid.close()

        best_param_index = np.argmax(grid_results)
        best_params = param_combinations[best_param_index]
        best_accuracy = grid_results[best_param_index]

        print("Best Parameters:", dict(zip(param_grid.keys(), best_params)))
        print("Best Accuracy:", best_accuracy)

        # Update the best model with the best parameters
        best_model.set_params(**dict(zip(param_grid.keys(), best_params)))
        serialized_model = pickle.dumps(best_model)

        return best_accuracy, accuracies, best_model, best_features

    print("Mean Accuracy:", mean_accuracy)

    return mean_accuracy, accuracies, best_model, best_features

import csv

import csv

def write_results_to_file(results, results_file):
    """
    Write the feature elimination results to a file.

    Args:
        results (list): List of dictionaries containing the feature elimination results.
        results_file (str): The file path to save the results.
    """
    with open(results_file+'.csv', 'w', newline='') as csvfile:
        fieldnames = ['Feature Size', 'Accuracy', 'Features','Best Features', 'Best Parameters']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def evaluate_feature_lists(model, param_grid, feature_sizes, x_train, y_train, x_test, y_test, step=1, gridsearch=True, results_file=None, feature_list = None):
    """
    Evaluate the model using different feature subsets and select the best subset based on feature importances.

    Args:
        model (object): The model to evaluate.
        param_grid (dict): The parameter grid for grid search (optional).
        feature_sizes (list): The desired feature sizes to evaluate.
        x_train (pandas.DataFrame): The feature matrix of the training set.
        y_train (pandas.Series or numpy.ndarray): The target vector of the training set.
        x_test (pandas.DataFrame): The feature matrix of the test set.
        y_test (pandas.Series or numpy.ndarray): The target vector of the test set.
        step (int): The number of features to remove in each iteration. Default is 1.
        gridsearch (bool): Whether to perform grid search. Default is True.
        results_file (str): The file path to save the results (optional).

    Returns:
        float: Mean performance of grid search (if gridsearch is True and param_grid is provided),
               otherwise None.
        list: List of accuracy scores for each feature subset.
        object: The best model selected based on the highest accuracy score (if gridsearch is True),
                otherwise None.
        list: The best features selected by the model as a list of feature names (if gridsearch is False),
              otherwise None.
    """
    if step <= 0:
        raise ValueError("Step value must be greater than 0.")

    accuracies = []
    best_model = None
    best_accuracy = 0.0
    best_features = None

    num_features = x_train.shape[1]
    if feature_list != None:
        all_features = feature_list
    else:
        all_features = x_train.columns.tolist()

    results = []

    for k in tqdm(feature_sizes, desc="Feature Sizes"):
        remaining_features = all_features.copy()

        p_bar = tqdm(total=len(remaining_features), desc="Feature Elimination", leave=False, postfix="")

        while len(remaining_features) >= k:
            p_bar.update(step)

            # Update feature matrix with remaining features
            x_train_selected = x_train[remaining_features]
            x_test_selected = x_test[remaining_features]

            # Fit the model on the training set and predict on the test set
            model.fit(x_train_selected, y_train)
            y_pred = model.predict(x_test_selected)

            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)

            # Save the best features and model if it has the highest accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_features = remaining_features.copy()
                best_model = model
                print('ITERATION BF',best_features)

            accuracies.append(accuracy)

            if len(remaining_features) <= k:
                print('Remaining Features /n',remaining_features)
                break

            # Get feature importances and eliminate the least important features
            importances = model.feature_importances_
            least_important_feature_indices = np.argsort(importances)[:step]
            least_important_features = [remaining_features[i] for i in least_important_feature_indices]

            for feature in least_important_features:
                print("Removed Feature:", feature )
                remaining_features.remove(feature)

            # Update the progress bar postfix with accuracies for each iteration
            pbar_postfix = f"Accuracies: {accuracies[-k:]}, Mean: {np.mean(accuracies[-k:])}, Features: {remaining_features}"
            p_bar.set_postfix_str(pbar_postfix)
            p_bar.total = len(remaining_features)
            p_bar.refresh()

        p_bar.close()

        # Append the current result to the results list
        results.append({
            'Feature Size': k,
            'Accuracy': best_accuracy,
            'Best Features': best_features,
            'Features': remaining_features
        })
        print('COMPLETED',best_features)

        if results_file is not None:
            # Write the results to the file after each process finishes
            write_results_to_file(results, results_file)

    mean_accuracy = np.mean(accuracies)

    if gridsearch:
        print("Performing Grid Search for Model:", model.__class__.__name__)
        print("Features:", best_features)
        print("Accuracies:", accuracies)
        print('Mean Accuracy:', mean_accuracy)

        # Perform grid search with 10-fold cross-validation
        grid_results = []
        param_combinations = list(product(*param_grid.values()))
        total_combinations = len(param_combinations)
        pbar_grid = tqdm(total=total_combinations, desc="Grid Search")

        for params in param_combinations:
            pbar_grid.set_postfix_str(f"Params: {params}")
            best_model.set_params(**dict(zip(param_grid.keys(), params)))

            # Update feature matrix with best features
            x_train_selected = x_train[best_features]
            x_test_selected = x_test[best_features]

            # Fit the model on the training set and predict on the test set
            model.fit(x_train_selected, y_train)
            y_pred = model.predict(x_test_selected)

            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            grid_results.append(accuracy)

            pbar_grid.update(1)

        pbar_grid.close()

        best_param_index = np.argmax(grid_results)
        best_params = param_combinations[best_param_index]
        best_accuracy = grid_results[best_param_index]

        print("Best Parameters:", dict(zip(param_grid.keys(), best_params)))
        print("Best Accuracy:", best_accuracy)

        # Update the best model with the best parameters
        best_model.set_params(**dict(zip(param_grid.keys(), best_params)))

        # Append the best parameters and accuracy to the results list
        results.append({
            'Feature Size': 'Grid Search',
            'Accuracy': best_accuracy,
            'Features': best_features.copy(),
            'Best Parameters': dict(zip(param_grid.keys(), best_params))
        })

        if results_file is not None:
            # Write the results to the file after the grid search finishes
            write_results_to_file(results, results_file)

    print("Mean Accuracy:", mean_accuracy)

    return mean_accuracy, accuracies, best_model, best_features

def evaluate_models(models, feature_sizes, x_train, y_train, x_test, y_test, params=None, step = 1, results_file = 'model/data/17/results', feature_list = None):
    """
    Evaluate multiple models using different feature subsets and parameter grid (if provided).

    Args:
        models (dict): Dictionary of model names and corresponding model objects.
        params (dict or None): The parameter grid for grid search (None if not performing grid search).
        feature_sizes (list): List of feature subsets to evaluate.
        x_train (pandas.DataFrame): The feature matrix of the training set.
        y_train (pandas.Series or numpy.ndarray): The target vector of the training set.
        x_test (pandas.DataFrame): The feature matrix of the test set.
        y_test (pandas.Series or numpy.ndarray): The target vector of the test set.

    Returns:
        dict: Dictionary of model names and corresponding lists of accuracy scores, best features, and mean accuracy.
    """
    results = {}
    for model_name, model in models.items():
        print(f"Evaluating model: {model_name}")
        # Set the feature size for each subset
        mean_accuracy, accuracies, best_model, best_features = evaluate_feature_lists(
            model, params, feature_sizes, x_train, y_train, x_test, y_test, step=step, results_file=results_file, feature_list = feature_list
        )

        results[model_name] = {
            'accuracies': accuracies,
            'best_model': model_name,
            'best_features': best_features,
            'mean_accuracy': mean_accuracy
        }

        if best_model is not None:
            # Save the best model if it has the highest accuracy
            save_best_model(best_model, f"{model_name}_best_model")

    return results

def get_metrics(y_true, y_pred):
    """
    Calculate various evaluation metrics.

    Args:
        y_true (numpy.ndarray): The true labels.
        y_pred (numpy.ndarray): The predicted labels.

    Returns:
        dict: Dictionary containing the evaluation metrics.
    """
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    fbeta = fbeta_score(y_true, y_pred, beta=1)
    roc_auc = roc_auc_score(y_true, y_pred)

    metrics = {
        "confusion_matrix": cm.tolist(),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "fbeta_score": fbeta,
        "roc_auc_score": roc_auc
    }

    return metrics

def print_metrics(y_true, y_pred, feature_names=None):
    """
    Print and visualize various evaluation metrics.

    Args:
        y_true (numpy.ndarray): The true labels.
        y_pred (numpy.ndarray): The predicted labels.
        feature_names (list or None): The names of the features (optional).

    Returns:
        None
    """
    metrics = get_metrics(y_true, y_pred)

    print("Confusion Matrix:")
    print(metrics['confusion_matrix'])
    print("Accuracy:", metrics['accuracy'])
    print("Precision:", metrics['precision'])
    print("Recall:", metrics['recall'])
    print("F-beta Score:", metrics['fbeta_score'])
    print("ROC AUC Score:", metrics['roc_auc_score'])

    if feature_names:
        print("Selected Features:", feature_names)

    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend()
    plt.show()

def save_best_model(model, filename):
    """
    Save the best model to a file.

    Args:
        model (object): The best model.
        filename (str): The filename to save the model.
    Returns:
        None
    """
    filename = "model/data/" + filename + ".pkl"
    with open(filename, 'wb') as file:
        joblib.dump(model, file)
    print("Best model saved as", filename)

def save_data_as_csv(x_train, y_train, x_test, y_test):
    """
    Save the train and test data, along with the best features, as separate CSV files.

    Args:
        features (list): List of best features.
        x_train (pandas.DataFrame): Training feature matrix.
        y_train (pandas.Series or numpy.ndarray): Training target vector.
        x_test (pandas.DataFrame): Test feature matrix.
        y_test (pandas.Series or numpy.ndarray): Test target vector.
    """

    # Save x_train and y_train as a single CSV file
    train_data = pd.concat([x_train, pd.DataFrame(y_train, columns=['churn'])], axis=1)
    train_data.to_csv('train_data.csv', index=False)

    # Save x_test and y_test as a single CSV file
    test_data = pd.concat([x_test, pd.DataFrame(y_test, columns=['churn'])], axis=1)
    test_data.to_csv('test_data.csv', index=False)


def load_new(save = True):
    df = load_dataset(split=False)
    df = generate_features(df, reduce_collinearity=True, threshold=0.6, heatmap=False)
    print_corr(df)
    
    df = preprocess_numeric_features(df)
    df.to_csv('model/data/added_feats_new_21.csv', index=False)

    df, x_train, x_test, y_train, y_test = transform(df)
    print(df, x_train, x_test, y_train, y_test)
    print(type(df), type(x_train), type(x_test), type(y_train), type(y_test))
    return  df, x_train, x_test, y_train, y_test 

def main():
    feats = load_feats()

    #feat_done = len(feats) == 0
    feat_done = True

    print(feats)
    if feat_done:
        df = pd.read_csv('model/data/added_feats_new.csv')
        df, x_train, x_test, y_train, y_test = transform(df)
    # Run the BaggingClassifier with RFECV
    else:
        # Load and preprocess the data
        df, x_train, x_test, y_train, y_test = load_new()

    results = rfa_bagging_rfecv(
        df,
        x_train,
        x_test,
        y_train,
        y_test,
        param_grid = {
        'n_estimators': [100,400,700,1000],
        'max_depth': [5,10,15,30, None],
        },
        base_classifier= RandomForestClassifier(criterion='entropy', max_depth=15, max_features=None, min_samples_leaf=1, min_samples_split=2, n_estimators=400,random_state=42),
        perform = ['rfecv'],                                                                                                                                                                                          
        cv=10,
        step=3,
        feat_done = False,
        results_file='model/data/17/results_for_each_iter'
    )
    print('Results:', results)


if __name__ == "__main__":
    main()
