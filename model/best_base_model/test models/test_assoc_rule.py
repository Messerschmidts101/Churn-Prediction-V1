import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def convert_numeric_literals(df):
    numeric_literal_cols = df.select_dtypes(include='object').apply(lambda x: pd.to_numeric(x, errors='ignore')).columns
    df[numeric_literal_cols] = df[numeric_literal_cols].apply(pd.to_numeric, errors='coerce')
    print('CONVERT',df)
    return df

def perform_binning(df, num_bins=5):
    numeric_cols = df.select_dtypes(include='number').columns
    discretizer = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy='uniform')
    df_numeric = df[numeric_cols].copy()
    imputer = SimpleImputer(strategy='mean')
    df_numeric_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=numeric_cols)
    df_numeric_imputed.index = df_numeric.index  # Reassign the index
    df_numeric_binned = pd.DataFrame(discretizer.fit_transform(df_numeric_imputed), columns=numeric_cols)
    return df_numeric_binned

def perform_label_encoding(df):
    df_encoded = df.copy()
    label_encoder = LabelEncoder()
    for col in df_encoded.select_dtypes(include='object').columns:
        df_encoded[col] = label_encoder.fit_transform(df_encoded[col])
    return df_encoded

def preprocess_data(df, num_bins=5):
    df_processed = convert_numeric_literals(df)
    df_binned = perform_binning(df_processed, num_bins=num_bins)
    df_encoded = perform_label_encoding(df_processed)
    df_final = pd.concat([df_encoded, df_binned], axis=1)
    return df_final

def split_data(df):
    X = df.drop('churn', axis=1)
    y = df['churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def perform_association_rule_mining(df):
    frequent_itemsets = apriori(df, min_support=0.05, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
    return rules

def main():
    df = pd.read_csv('model/data/train.csv')

    if 'churn' not in df.columns:
        print("The 'churn' column is not present in the DataFrame.")
        return

    df_processed = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df_processed)
    rules = perform_association_rule_mining(df_processed)

    print("Association Rules:")
    print(rules)

if __name__ == '__main__':
    main()