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


def rfa_bagging_rfecv(x_train, x_test, y_train, y_test, param_grid=None, base_classifier=None, bagging_params=None, cv=5, step=10):
    
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
    '''
    # Train the base RandomForestClassifier with grid search
    grid_search = GridSearchCV(estimator=base_classifier, param_grid=param_grid, cv=cv, verbose=2)

    param_combinations = list(product(param_grid['n_estimators'], param_grid['max_depth']))
    total_iterations = len(param_combinations)

    with tqdm(total=total_iterations, desc="Grid Search") as pbar:
        for estimator, depth in param_combinations:
            base_classifier = RandomForestClassifier(
                n_estimators=estimator,
                max_depth=depth,
                class_weight=class_weights,
                random_state=42
            )
            base_classifier.fit(x_train, y_train)
            accuracy, y_pred = evaluate_model(base_classifier, selected_indices, x_test, y_test)
            pbar.set_postfix({'Best_Params': grid_search.best_params_, 'Best_Score': grid_search.best_score_, 'Mean_score': accuracy})
            pbar.update(1)
    best_base_classifier = grid_search.best_estimator_
    # Train the best base classifier
    best_base_classifier.fit(x_train_selected, y_train)

    # Save the best estimator
    joblib.dump(best_base_classifier, 'best_estimator.pkl')'''
    
    best_base_classifier = RandomForestClassifier(criterion='entropy', max_depth=15, max_features=None, min_samples_leaf=1, min_samples_split=2, n_estimators=100,class_weight=dict(enumerate(class_weights)),random_state=42)

    '''
    Grid Search: 486it [7:55:22, 58.69s/it, Best_Params={'criterion': 'entropy', 'max_depth': 15, 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}, Best_Score=0.979, Mean_score=0.978]
    '''

    # Perform RFECV to select the most relevant features
    with tqdm(total=1, desc="Feature Selection") as pbar:
        selector = RFECV(estimator=best_base_classifier, step=step, cv=StratifiedKFold(cv), scoring='accuracy', verbose=2)
        selector.fit(x_train, y_train)
        x_train_selected = selector.transform(x_train)
        selected_indices = selector.support_

        # Get the feature names corresponding to the selected indices
        feature_names = x_train.columns[selected_indices]

        # Save the feature names to a file
        save_to_json(feature_names.tolist())

        pbar.update(1)
    
    ada_model = AdaBoostClassifier(base_estimator=best_base_classifier, n_estimators=100)
    ada_model.fit(x_train_selected, y_train)

    # Train the BaggingClassifier
    '''bagging_classifier = train_bagging_classifier(
        best_base_classifier,
        x_train_selected,
        y_train,
        n_estimators=bagging_params['n_estimators'],
        max_samples=bagging_params['max_samples'],
        max_features=bagging_params['max_features'],
        random_state=bagging_params['random_state']
    )'''
    # Create a list of models and their names
    models = [
        ('Base Model', best_base_classifier),
        ('AdaBoost Model', ada_model)
    ]

    # Evaluate and print metrics for each model
    with tqdm(total=len(models), desc="Evaluation") as pbar:
        accuracies = []
        for name, model in models:
            accuracy, y_pred = evaluate_model(model, selected_indices, x_test, y_test)
            pbar.set_postfix({'Accuracy': accuracy})
            pbar.update(1)

            # Print the accuracy for the current model
            print(f"Accuracy ({name}): {accuracy}")

            # Print and save the metrics for the current model
            print(f"Metrics for {name}:")
            print_metrics(y_test, y_pred)
            save_best_model(model, accuracy)

            accuracies.append(accuracy)

    # Return the list of accuracies
    return accuracies

def evaluate_model(model, selected_indices, feature_names, x_test, y_test):
    x_test_selected = pd.DataFrame(x_test[:, selected_indices], columns=feature_names)
    y_pred = model.predict(x_test_selected)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, y_pred

def print_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    fbeta = fbeta_score(y_true, y_pred, beta=1)
    roc_auc = roc_auc_score(y_true, y_pred)

    print("Confusion Matrix:")
    print(cm)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F-beta Score:", fbeta)
    print("ROC AUC Score:", roc_auc)

    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend()
    plt.show()

def save_best_model(model, accuracy):
    best_model_filename = "best_model.pkl"
    joblib.dump(model, best_model_filename)
    print("Best model saved as", best_model_filename)

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
