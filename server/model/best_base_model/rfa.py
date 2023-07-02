import csv
from itertools import product
import json
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

def rfa_bagging_rfecv(df, x_train, x_test, y_train, y_test, param_grid=None, base_classifier=None, perform = ['grid_search', 'rfecv'], cv=5, step=10):
    
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
    valid = x_train.columns.equals(df.columns)
    print(x_train.shape[1], df.shape[1])
    print("VALIDATOR",valid)
    
    if valid:
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
            pbar.update(1)
    # Create a list of models and their names
        models = [
            ('Base Model', best_base_classifier),
        ]

        ada_model = AdaBoostClassifier(base_estimator=best_base_classifier, n_estimators=400)
        ada_model.fit(x_train_selected, y_train)
        models.append(('AdaBoost Model', ada_model))
        
    # Define your parameter grid
        feature_lists = load_feats(top_list=True)
        results = evaluate_models(df, { "rfa": best_base_classifier }, feature_lists, x_train, y_train, x_test, y_test, params=param_grid)

        return results


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from tqdm import tqdm

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

def evaluate_feature_lists(model, param_grid, feature_sizes, x_train, y_train, x_test, y_test):
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

            if num_features - feature_size >= 5:
                # Select top features based on step size (5)
                selected_indices = sorted_indices[:num_features - 5]
            else:
                # Select top features based on feature_size
                selected_indices = sorted_indices[:feature_size]

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
            print(f"Evaluated Features (Feature Size: {num_features}): {selected_features}")
            print_metrics(y_test, y_pred)

            # Update the feature matrix and feature names for the next iteration
            x_train = x_train_selected
            x_test = x_test_selected
            all_features = selected_features
            num_features = feature_size

    return accuracies, best_model, best_features

def evaluate_models(df, models, feature_sizes, x_train, y_train, x_test, y_test, params = None,):
    """
    Evaluate multiple models using different feature subsets and parameter grid (if provided).

    Args:
        models (dict): Dictionary of model names and corresponding model objects.
        params (dict or None): The parameter grid for grid search (None if not performing grid search).
        feature_lists (list): List of feature subsets to evaluate.
        x_train (numpy.ndarray): The feature matrix of the training set.
        y_train (numpy.ndarray): The target vector of the training set.
        x_test (numpy.ndarray): The feature matrix of the test set.
        y_test (numpy.ndarray): The target vector of the test set.

    Returns:
        dict: Dictionary of model names and corresponding lists of accuracy scores.
    """
    results = {}
    for model_name, model in models.items():
        print(f"Evaluating model: {model_name}")
        # Set the feature size for each subset
        feature_sizes = [50,45,40, 35, 30]
        accuracy, best_model, best_features = evaluate_feature_lists(model, params, feature_sizes, x_train, y_train, x_test, y_test)

        if best_model is not None:
            # Save the best model if it has the highest accuracy
            save_best_model(best_model, f"{model_name}_best_model")

    return results
    metrics = print_metrics(y_true, y_pred)
    metrics_json = json.dumps(metrics, indent=4)
    print(metrics_json)

import json

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


def main():
    # Load and preprocess the data
    df = load_dataset(split=False)
    df = generate_features(df, reduce_collinearity=True)
    print_corr(df)
    
    df = preprocess_numeric_features(df)
    df.to_csv('model/data/added_feats.csv', index=False)

    df, x_train, x_test, y_train, y_test = transform(df)
    print(df, x_train, x_test, y_train, y_test)
    print(type(df), type(x_train), type(x_test), type(y_train), type(y_test))
    
    #class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)

    # Run the BaggingClassifier with RFECV
    accuracies = rfa_bagging_rfecv(
        df,
        x_train,
        x_test,
        y_train,
        y_test,
        param_grid = {
        'n_estimators': [400,600,800,1000],
        'max_depth': [15,20,30, None],
        },
        base_classifier= RandomForestClassifier(criterion='entropy', max_depth=15, max_features=None, min_samples_leaf=1, min_samples_split=2, n_estimators=100,random_state=42),
        perform = ['rfecv'],
        cv=10,
        step=3
    )
    print('Accuracies', accuracies)

if __name__ == "__main__":
    main()
