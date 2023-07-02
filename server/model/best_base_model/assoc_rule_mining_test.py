import numpy as np
from dataset import load_dataset
import discretizer as dc
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

def find_itemsets(df, filter_condition, column_names, min_support=0.1):
    """
    Find frequent itemsets in the DataFrame based on a filter condition and selected columns.

    Parameters:
        df (DataFrame): The input DataFrame.
        filter_condition (str): The filter condition to select specific rows from the DataFrame.
        column_names (str or list): The name(s) of the column(s) to use for generating itemsets.
        min_support (float): The minimum support threshold for itemsets. Default is 0.1.

    Returns:
        DataFrame: The DataFrame containing frequent itemsets with their support values.
    """
    # Filter the DataFrame based on the provided condition and selected columns
    filtered_df = df.query(filter_condition)[column_names]

    # Convert the selected columns to a list of lists for transactions
    if isinstance(column_names, str):
        transactions = filtered_df.apply(lambda x: [x]).tolist()
    else:
        transactions = filtered_df.values.tolist()

    # Perform one-hot encoding of the transactions
    te = TransactionEncoder()
    te_ary = te.fit_transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    # Apply the Apriori algorithm
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)

    return frequent_itemsets


def assign_frequent_itemsets(df, frequent_itemsets, score_column='Score'):
    """
    Assigns frequent itemsets to the corresponding rows in a DataFrame and adds a score column for each itemset.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        frequent_itemsets (pd.DataFrame): The DataFrame containing frequent itemsets and their support scores.
        score_column (str): The name of the column to create in the input DataFrame for storing the support scores.
                            Default is 'Score'.
    """
    for itemset in frequent_itemsets['itemsets']:
        itemset_str = ' '.join(str(item) for item in itemset)
        df[itemset_str] = np.where(df['Assigned_Itemsets'] == itemset_str, frequent_itemsets.loc[
            frequent_itemsets['itemsets'] == itemset, 'support'].values[0], 0)

    return df


def main():
    df= load_dataset(transformed=True,split=False)
    print(df)
    selected = dc.select_col_to_transform(df,'int32',2)
    print(df[selected].head(2))
    selected.remove('churn')
    print(selected)
    frequent_itemsets_1 = find_itemsets(df, "churn == 1", df.columns, min_support=0.1)
    frequent_itemsets_0 = find_itemsets(df, "churn == 0", df.columns, min_support=0.1)

    initial_columns = df.columns
    frequent_itemsets = frequent_itemsets_0.merge(frequent_itemsets_1)
    print(frequent_itemsets)
    df['Assigned_Itemsets'] = df.apply(lambda row: ' '.join([str(col) for col in frequent_itemsets['itemsets'].values if col != 'support' and row[col] == 1]), axis=1)

    df = assign_frequent_itemsets(df, frequent_itemsets_1, score_column='Score_1')
    df = assign_frequent_itemsets(df, frequent_itemsets_0, score_column='Score_0')

    # Display the updated DataFrame with frequent itemsets assigned
    added_columns = df.columns.difference(initial_columns)
    filtered_df = df.loc[(df[added_columns] == 1).any(axis=1)]
    print('Filtered',filtered_df)
    '''
    What should we do with this data?
        - Calculate the difference of support for similar itemsets
        - Col has value of 0(-.4) where churn = 1,
        - If 1(+20) where churn = 1
         support              itemsets
    0   0.963211                 (0.0)
    1   1.000000                 (1.0)
    2   0.526756                 (2.0)
    3   0.267559                 (3.0)
    4   0.306020                 (4.0)
    5   0.205686                 (5.0)
    6   0.140468                 (6.0)
    7   0.963211            (0.0, 1.0)
    8   0.508361            (0.0, 2.0)
    9   0.255853            (0.0, 3.0)
    10  0.301003            (0.0, 4.0)
    11  0.198997            (0.0, 5.0)
    12  0.133779            (0.0, 6.0)
    13  0.526756            (1.0, 2.0)
    14  0.267559            (1.0, 3.0)
    15  0.306020            (1.0, 4.0)
    16  0.205686            (1.0, 5.0)
    17  0.140468            (1.0, 6.0)
    18  0.128763            (2.0, 3.0)
    19  0.143813            (2.0, 4.0)
    20  0.508361       (0.0, 1.0, 2.0)
    21  0.255853       (0.0, 1.0, 3.0)
    22  0.301003       (0.0, 1.0, 4.0)
    23  0.198997       (0.0, 1.0, 5.0)
    24  0.133779       (0.0, 1.0, 6.0)
    25  0.125418       (0.0, 2.0, 3.0)
    26  0.140468       (0.0, 2.0, 4.0)
    27  0.128763       (1.0, 2.0, 3.0)
    28  0.143813       (1.0, 2.0, 4.0)
    29  0.125418  (0.0, 1.0, 2.0, 3.0)
    30  0.140468  (0.0, 1.0, 2.0, 4.0)      support         itemsets
    0   1.000000            (0.0)
    1   0.802574            (1.0)
    2   0.512596            (2.0)
    3   0.340361            (3.0)
    4   0.228368            (4.0)
    5   0.173877            (5.0)
    6   0.122946            (6.0)
    7   0.802574       (0.0, 1.0)
    8   0.512596       (0.0, 2.0)
    9   0.340361       (0.0, 3.0)
    10  0.228368       (0.0, 4.0)
    11  0.173877       (0.0, 5.0)
    12  0.122946       (0.0, 6.0)
    13  0.367744       (1.0, 2.0)
    14  0.253012       (1.0, 3.0)
    15  0.174699       (1.0, 4.0)
    16  0.134173       (1.0, 5.0)
    17  0.151424       (2.0, 3.0)
    18  0.367744  (0.0, 1.0, 2.0)
    19  0.253012  (0.0, 1.0, 3.0)
    20  0.174699  (0.0, 1.0, 4.0)
    21  0.134173  (0.0, 1.0, 5.0)
    22  0.151424  (0.0, 2.0, 3.0)
    '''
if __name__ == '__main__':
    main()