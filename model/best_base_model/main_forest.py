import numpy as np
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from dataset import load_dataset
from gen_feat import generate_features, print_corr
from rfa_final import rfa_bagging_rfecv

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


def preprocess_numeric_features(x_train, x_test):
    # Select only the numeric features
    x_train_numeric = x_train.select_dtypes(include='number')
    x_test_numeric = x_test.select_dtypes(include='number')

    # Replace infinity and very large values with a finite number
    x_train_numeric = np.nan_to_num(x_train_numeric, nan=0, posinf=1e12, neginf=-1e12)
    x_test_numeric = np.nan_to_num(x_test_numeric, nan=0, posinf=1e12, neginf=-1e12)

    return x_train_numeric, x_test_numeric

def run():
    df = load_dataset(split=False)
    print(df)
    df = generate_features(df)
    print_corr(df)
    
    df, x_train, x_test, y_train, y_test = transform(df)
    # Select only the numeric features
    x_train_numeric, x_test_numeric = preprocess_numeric_features(x_train, x_test)
    #Training and Evaluation
    accuracy = rfa_bagging_rfecv(df, x_train_numeric, x_test_numeric, y_train, y_test, step=5, cv =10)
    # Print the accuracy
    print("Accuracy:", accuracy)

def main():
    run()
if __name__ == '__main__':
    main()