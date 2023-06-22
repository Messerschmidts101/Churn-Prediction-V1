import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    fbeta_score,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import joblib
import csv
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_curve, fbeta_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.utils import compute_class_weight
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

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

def load_data():
    # Load your data here
    x = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test

def train_base_classifier(x_train, y_train, class_weights, base_classifier=None):
    if base_classifier is None:
        base_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight=class_weights,
            random_state=42
        )
    base_classifier.fit(x_train, y_train)
    return base_classifier

def train_bagging_classifier(base_classifier, x_train_selected, y_train, n_estimators=10, max_samples=0.8, max_features=0.8, random_state=42):
    bagging_classifier = BaggingClassifier(
        base_estimator=base_classifier,
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        random_state=random_state
    )
    bagging_classifier.fit(x_train_selected, y_train)
    return bagging_classifier

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

def rfa_bagging_rfecv(df, x_train, x_test, y_train, y_test, param_grid=None, base_classifier=None, bagging_params=None, cv=5, step=10):
    
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
        base_classifier = RandomForestClassifier(class_weight=class_weights, random_state=42)

    if bagging_params is None:
        bagging_params = {
        'n_estimators': [5, 10, 15],
        'max_samples': [0.6, 0.7, 0.8],
        'max_features': [0.6, 0.7, 0.8],
        'bootstrap': [True, False],
        'bootstrap_features': [True, False],
        'oob_score': [True, False],
        'warm_start': [True, False],
        'random_state': [42]
    }

    
    best_base_classifier = RandomForestClassifier(criterion='entropy', max_depth=15, max_features=None, min_samples_leaf=1, min_samples_split=2, n_estimators=100,class_weight=dict(enumerate(class_weights)),random_state=42)
    # Create a list of models and their names
    models = [
        ('Base Model', best_base_classifier),
        #('AdaBoost Model', ada_model)
    ]
   # Define your parameter grid
    feature_lists = load_feats(top_list=True)
    results = evaluate_models(df, { "rfa": best_base_classifier }, None, x_train, y_train, x_test, y_test, None)
    return results

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

def evaluate_feature_lists(model, param_grid, feature_sizes, df, x_train, y_train, x_test, y_test,step = 5):
    """
    Evaluate the model using different feature subsets and select the best subset based on feature importances.

    Args:
        model (object): The model to evaluate.
        param_grid (dict): The parameter grid for grid search (optional).
        feature_sizes (list): The desired feature sizes to evaluate.
        df (pandas.DataFrame): The dataframe containing the features (without the churn column).
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
    print(type(df))
    num_features = x_train.shape[1]
    all_features = df.columns.tolist()

    for feature_size in feature_sizes:
        while num_features > feature_size:
            # Train the model and get feature importances
            model.fit(x_train, y_train)
            feature_importances = model.feature_importances_

            # Sort feature importances in descending order
            sorted_indices = np.argsort(feature_importances)[::-1]

            if num_features - feature_size >= step:
                # Select top features based on step size (5)
                selected_indices = sorted_indices[:num_features - step]
            else:
                # Select top features based on feature_size
                selected_indices = sorted_indices[:feature_size]

            selected_features = [all_features[index] for index in selected_indices]

            print("SHAPE:",x_train.shape, df.shape)
            print(x_train.columns.equals(df.columns))
            print(x_train.columns.tolist() == df.columns.tolist())
            # Update feature matrix with selected features
            print(type(x_train), type(y_train))
            x_train_selected = x_train.iloc[:, selected_indices]
            x_test_selected = x_test.iloc[:, selected_indices]
           

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

            # Print the metrics for the current feature subset
            print(f"Feature Size: {feature_size}, Accuracy: {accuracy:.4f}")
            print_metrics(y_test, y_pred, selected_features)

            # Update the feature matrix and feature names for the next iteration
            x_train = x_train_selected
            x_test = x_test_selected
            all_features = selected_features
            num_features = feature_size

    return accuracies, best_model, best_features


def evaluate_models(df, models, feature_lists, x_train, y_train, x_test, y_test, params=None):
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
        feature_sizes = [40, 35, 30]
        accuracies, best_model, best_features = evaluate_feature_lists(model, params, feature_sizes, df.drop('churn', axis=1), x_train, y_train, x_test, y_test)

        if best_model is not None:
            # Save the best model if it has the highest accuracy
            save_best_model(best_model, f"{model_name}_best_model")

        results[model_name] = accuracies

    return results

def main():
    # Load and preprocess the data
    x_train, x_test, y_train, y_test = load_data()

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)

    # Run the BaggingClassifier with RFECV
    accuracies = rfa_bagging_rfecv(
        x_train,
        x_test,
        y_train,
        y_test,
        param_grid={
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15]
        },
        base_classifier=RandomForestClassifier(class_weight=class_weights, random_state=42),
        bagging_params={
            'n_estimators': 10,
            'max_samples': 0.8,
            'max_features': 0.8,
            'random_state': 42
        },
        cv=5,
        step=1
    )
    print('Accuracies', accuracies)

if __name__ == "__main__":
    main()
