from best_base_model.dataset import load_dataset
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def apriori_(df):
    
    transactions = prepare_data_for_apriori(df_encoded, binned_columns)
    frequent_itemsets = apriori(df, min_support=0.05, use_colnames=True)
    print(frequent_itemsets)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    #Access and analyze the generated rules
    print(rules.head())

def rule_mine(df ,method= apriori_):
    method(df)

def prepare_data_for_apriori(df, columns):
    transactions = []
    for i in range(len(df)):
        transaction = []
        for col in columns:
            transaction.append(str(df.loc[i, col]))
        transactions.append(transaction)
    return transactions

def main():
    df, x_train, x_test, y_train, y_test = load_dataset()

    
    '''print(df.head(10))
    for col in df.columns:
        if 35.0 in df[col].values:
            print(f"Column '{col}' contains the value 35.0")'''
    #rule_mine(df)


if __name__ == "__main__":
    main()