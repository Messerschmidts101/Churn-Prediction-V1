import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from dataset import load_dataset
from gen_feat import generate_features, print_corr
from rfa_x import rfa_bagging_rfecv

def transform(df):
    target = df['churn']
    features = df.drop('churn', axis=1)
    # Splitting the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    #print(features.columns)
    return df, x_train, x_test, y_train, y_test

def sig_dif(churned_accounts, non_churned_accounts):
    columns_of_interest = churned_accounts.columns[3:19]
    for column in columns_of_interest:
        churned_values = churned_accounts[column]
        non_churned_values = non_churned_accounts[column]

        t_statistic, p_value = ttest_ind(churned_values, non_churned_values)
        print(f"Column: {column}")
        print(f"T-Statistic: {t_statistic}")
        print(f"P-Value: {p_value}")
        print("\n")

def preprocess_numeric_features(df):
    # Select only the numeric features
    df = df.select_dtypes(include='number')
    column_names = df.columns.tolist()

    # Replace infinity and very large values with a finite number
    df = np.nan_to_num(df, nan=0, posinf=1e12, neginf=-1e12)

    # Convert the modified NumPy array back to a pandas DataFrame with original column names
    df = pd.DataFrame(df, columns=column_names)

    return df


def run():
    df = load_dataset(split=False)
    df = generate_features(df, reduce_collinearity=True)
    print_corr(df)
    
    df = preprocess_numeric_features(df)
    df.to_csv('model/data/added_feats.csv', index=False)

    df, x_train, x_test, y_train, y_test = transform(df)
    # Select only the numeric features
    print(df.shape, x_train.shape, x_test.shape, y_train, y_test.shape)

    #Training and Evaluation
    accuracy = rfa_bagging_rfecv(df, x_train, x_test, y_train, y_test, step=5, cv=10)

    # Print the accuracy
    print("Accuracy:", accuracy)

def main():
    run()
if __name__ == '__main__':
    main()