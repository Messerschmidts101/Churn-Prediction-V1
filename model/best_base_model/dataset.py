import numpy
import pandas as pd
from sklearn.calibration import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split


from sklearn.preprocessing import KBinsDiscretizer

def binning_and_encode(df, num_bins=5):
    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    for col in numeric_cols:
        df['binned_' + col] = pd.cut(df[col], bins=num_bins)

    df_encoded = pd.get_dummies(df, columns=numeric_cols, drop_first=True)

    return df_encoded


def binning_and_encode_(df, num_bins=5):
    # Select numeric columns for binning
    numeric_cols = df.select_dtypes(include='number').columns

    # Perform binning on numeric columns
    discretizer = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy='uniform')
    df_numeric_binned = pd.DataFrame(discretizer.fit_transform(df[numeric_cols]), columns=numeric_cols)

    # Perform one-hot encoding on the original and binned columns
    df_encoded = pd.get_dummies(pd.concat([df, df_numeric_binned], axis=1))

    return df_encoded
def binning_x(df, columns=None, num_bins=5, bin_labels=None, bin_ranges=None):
    if columns is None:
        columns = df.select_dtypes(include='number').columns.tolist()

    for col in columns:
        # Check if the column is numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            # If bin ranges are not provided, calculate them dynamically
            if bin_ranges is None:
                min_val = df[col].min()
                max_val = df[col].max()
                bin_width = (max_val - min_val) / num_bins
                bin_ranges = [(min_val + i * bin_width, min_val + (i + 1) * bin_width) for i in range(num_bins)]

            # Perform binning for numeric columns
            bin_labels = [f'[{left:.2f}, {right:.2f})' for left, right in bin_ranges]
            bin_intervals = pd.cut(df[col], [left for left, _ in bin_ranges] + [max_val], labels=bin_labels)
            df['binned_' + col] = bin_intervals

    return df.drop(columns=columns)

def binning(df, columns=None, num_bins=5):
    if columns is None:
        columns = df.select_dtypes(include='number').columns.tolist()

    for col in columns:
        # Check if the column is numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            # Fill missing values with the median of the column
            df[col].fillna(df[col].median(), inplace=True)

            # Perform binning for numeric columns
            bin_intervals = pd.cut(df[col], num_bins)
            binned_col_name = 'binned_' + col
            df[binned_col_name] = pd.qcut(df[col], num_bins, labels=False, duplicates='drop')
            df = pd.get_dummies(df, columns=[binned_col_name], prefix=binned_col_name, drop_first=True)
            df.drop(col, axis=1, inplace=True)  # Drop the original column
        else:
            # For non-numeric columns, check if it's bool
            if pd.api.types.is_bool_dtype(df[col]):
                # For bool columns, simply copy the column to the binned column
                df['binned_' + col] = df[col]
            else:
                # For other columns, convert them to bool (True/False) values
                df['binned_' + col] = df[col].astype(bool).astype(int)
                df.drop(col, axis=1, inplace=True)  # Drop the original column

    return df

def binning__(df, columns=None, num_bins=5):
    if columns is None:
        columns = df.select_dtypes(include='number').columns.tolist()

    for col in columns:
        # Check if the column is numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            # Perform binning for numeric columns
            bin_intervals = pd.cut(df[col], num_bins)
            bin_labels = [f'[{interval.left}, {interval.right})' for interval in bin_intervals]
            df['binned_' + col] = bin_labels
        else:
            # For non-numeric columns, check if it's bool
            if pd.api.types.is_bool_dtype(df[col]):
                # For bool columns, simply copy the column to the binned column
                df['binned_' + col] = df[col]
            else:
                # For other columns, convert them to bool (True/False) values
                df['binned_' + col] = df[col].astype(bool).astype(int)

    return df

def binning_(df, columns=None, num_bins=5, bin_labels=None):
    if columns is None:
        columns = df.select_dtypes(include='number').columns.tolist()

    for col in columns:
        # Check if the column is numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            # If bin labels are not provided, generate them dynamically
            if bin_labels is None:
                min_val = df[col].min()
                max_val = df[col].max()
                bin_width = (max_val - min_val) / num_bins
                bin_labels = [f'{min_val + i*bin_width:.2f}-{min_val + (i+1)*bin_width:.2f}' for i in range(num_bins)]

            # Perform binning for numeric columns
            bin_intervals = pd.cut(df[col], num_bins, labels=bin_labels)
            df['binned_' + col] = bin_intervals
        else:
            # For non-numeric columns, simply copy the column to the binned column
            df['binned_' + col] = df[col]

    return df
def binning_(df, columns=None, num_bins=5, bin_labels=None, remove_original=True):
    if columns is None:
        columns = df.select_dtypes(include='number').columns.tolist()

    for col in columns:
        # Check if the column is numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            # If bin labels are not provided, generate them dynamically
            if bin_labels is None:
                min_val = df[col].min()
                max_val = df[col].max()
                bin_width = (max_val - min_val) / num_bins
                bin_labels = [f'{min_val + i*bin_width:.2f}-{min_val + (i+1)*bin_width:.2f}' for i in range(num_bins)]

            # Perform binning for numeric columns
            bin_intervals = pd.cut(df[col], num_bins, labels=bin_labels)
            df['binned_' + col] = bin_intervals

            if remove_original:
                df.drop(col, axis=1, inplace=True)  # Drop the original column
        else:
            # For non-numeric columns, simply copy the column to the binned column
            df['binned_' + col] = df[col]

    return df

def transform(df, binned = False, split = True):
    '''
    Performs data transformation on the input DataFrame.
        Parameters:
        df (pd.DataFrame): Input DataFrame to be transformed.
        binned (bool): If True, applies binning transformation. Default is False.
        split (bool): If True, performs data splitting. Default is True.

        Returns:
            tuple: A tuple containing the following elements:
                - df: Processed dataset with encoded features
                - x_train: Training data features
                - x_test: Testing data features
                - y_train: Training data labels
                - y_test: Testing data labels
    '''

    if 'churn' not in df.columns:
        raise KeyError("'churn' column not found in the DataFrame.")

    # Perform one-hot encoding on categorical variables
    one_hot_encoder(df)
    target = df['churn']
    #SMOTE Samplter to balance the
    smote = SMOTE()
    x_resampled, y_resampled = smote.fit_resample(df[[col for col in df.columns if col !='churn']], target)
    # Convert the resampled data back into a DataFrame
    resampled_df = pd.DataFrame(x_resampled, columns=df.columns[:-1])  # Assuming the last column is the target column
    resampled_df['churn'] = y_resampled

    if binned:
        # Perform binning and convert numeric values to binary format
        df = binning_and_encode(df, num_bins=5)

    if split:
        #print(sorted(Counter(y_resampled).items()),y_resampled.shape)
        # Split the data into training and testing sets
        return df, *train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=42)

    return resampled_df




def one_hot_encoder(df):
    '''
    Converts the categorical values to numerical 
    '''
    label_encoder = LabelEncoder()

    # Encode categorical columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = label_encoder.fit_transform(df[col])

def print_desc(df):
    if 'churn' not in df.columns:
        print("The 'churn' column is not present in the DataFrame.")
        return

    # number of churn and non-churn 
    counts = df['churn'].value_counts()
    perc_churn = (counts[1] / (counts[0] + counts[1])) * 100

    # no. of duplicates 
    duplicates = len(df[df.duplicated()])

    # no of missing values
    missing_values = df.isnull().sum().sum()

    # Data types in dataset
    types = df.dtypes.value_counts()
    print("Churn Rate = %.1f %%"%(perc_churn))
    print('Number of Duplicate Entries: %d'%(duplicates))
    print('Number of Missing Values: %d'%(missing_values))
    print('Number of Features: %d'%(df.shape[1]))
    print('Number of Customers: %d'%(df.shape[0]))
    print('Data Types and Frequency in Dataset:')
    print(types)



import pandas as pd
import numpy as np


def load_dataset(desc=False, bin=False, transformed=True, split=True) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Loads the dataset, performs data exploration, and returns the processed dataset and data splits.
        Parameters:
        desc (bool): If True, prints data exploration summary. Default is False.
        bin (bool): If True, applies binning transformation. Default is False.
        transformed (bool): If True, returns the transformed dataset. Default is True.
        split (bool): If True, performs data splitting. Default is True.

    Returns:
        tuple: A tuple containing the following elements:
            - df: Processed dataset with encoded features
            - x_train: Training data features
            - x_test: Testing data features
            - y_train: Training data labels
            - y_test: Testing data labels
    '''

    df = pd.read_csv('model/data/train.csv')

    if desc:
        print_desc(df)
    if transformed == True:
        return transform(df, bin, split)
    return df


def main():
    import numpy as np
    df, x_train, x_test, y_train, y_test  =  load_dataset()
    print(df['churn'])

    #df = pd.read_csv('model/data/train.csv')
    #print(df.select_dtypes(include='object').columns)
if __name__ == '__main__':
    main()