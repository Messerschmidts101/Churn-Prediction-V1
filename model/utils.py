import pandas as pd
import numpy as np

def calculate_num_bins(df, columns, method='range', k=1.0):
    """
    Calculates the optimal number of bins for dynamic binning based on the range or standard deviation of columns in a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        columns (list): The list of column names.
        method (str, optional): The method to use for determining the number of bins.
            - 'range': Based on the range of values (default).
            - 'std_dev': Based on the standard deviation of values.
        k (float, optional): The scaling factor for the standard deviation method. Ignored if method is 'range'.

    Returns:
        dict: A dictionary mapping each column name to its optimal number of bins.

    """
    num_bins_dict = {}
    
    for column in columns:
        values = df[column].dropna().values
        if method == 'range':
            range_val = np.max(values) - np.min(values)
            estimated_bin_width = range_val / np.sqrt(len(values))
            num_bins = int(np.ceil(range_val / estimated_bin_width))
        elif method == 'std_dev':
            std_dev = np.std(values)
            estimated_bin_width = k * std_dev
            num_bins = int(np.ceil(std_dev / estimated_bin_width))
        else:
            raise ValueError("Invalid method. Supported methods are 'range' and 'std_dev'.")
        
        num_bins_dict[column] = num_bins
    
    return num_bins_dict

# Example usage
data = {'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
df = pd.DataFrame(data)

num_bins_range = calculate_num_bins(df, 'A', method='range')
print("Number of bins (range method):", num_bins_range)

num_bins_std_dev = calculate_num_bins(df, 'A', method='std_dev')
print("Number of bins (std_dev method):", num_bins_std_dev)
