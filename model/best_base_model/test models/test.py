import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

def dynamic_binning(df, column, num_bins):
    min_val = df[column].min()
    max_val = df[column].max()
    bin_width = (max_val - min_val) / num_bins
    bin_edges = [min_val + i * bin_width for i in range(num_bins + 1)]
    binned_column = pd.cut(df[column], bins=bin_edges, include_lowest=True)
    return binned_column

def one_hot_encode(transactions):
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_array, columns=te.columns_)
    return df_encoded

def apriori_analysis(df_encoded, min_support):
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    return frequent_itemsets

def preprocess_dataset(df, num_bins=10, min_support=0.1):
    transformed_data = []
    
    for column in df.columns:
        # Perform dynamic binning on each column
        binned_column = dynamic_binning(df, column, num_bins)
        
        # Convert binned column to a list of transactions
        transactions = [[str(row)] for row in binned_column]
        
        # Add the transactions to the transformed data
        transformed_data.extend(transactions)
    
    # Apply one-hot encoding to the transaction data
    df_encoded = one_hot_encode(transformed_data)
    
    # Apply Apriori analysis
    frequent_itemsets = apriori_analysis(df_encoded, min_support)
    
    return frequent_itemsets

def main():
    # Load your dataset
    df = pd.read_csv('model/data/train.csv')
    
    # Preprocess the dataset and perform Apriori analysis
    frequent_itemsets = preprocess_dataset(df, num_bins=10, min_support=0.1)
    
    # Print the frequent itemsets
    print(frequent_itemsets)
    
if __name__ == '__main__':
    main()
