import pandas as pd
from sklearn.metrics import confusion_matrix
from dataset import load_dataset
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def add_cluster_labels(df, columns_of_interest, num_clusters):
    """
    Perform K-means clustering on the selected columns of a DataFrame and add a column with descriptive cluster labels.

    Parameters:
        df (DataFrame): DataFrame to perform clustering on.
        columns_of_interest (list): List of column names to use for clustering.
        num_clusters (int): Number of clusters to generate.

    Returns:
        df_clustered (DataFrame): DataFrame with an additional column of cluster labels.
    """
    cluster_labels = KMeans(n_clusters=num_clusters, random_state=0).fit_predict(df[columns_of_interest])
    new_column_name = 'cluster_labels_' + '_'.join(columns_of_interest)
    df_clustered = pd.concat([df, pd.DataFrame({new_column_name: cluster_labels})], axis=1)
    return df_clustered


def check_plot(df, x, y):
    """
    Generate a KDE plot for the specified columns in the DataFrame.

    Parameters:
        df (DataFrame): DataFrame containing the data.
        x (str): Name of the column to plot on the x-axis.
        y (str): Name of the column to plot on the y-axis.
    """
    sns.kdeplot(data=df, x=x, y=y, n_levels=20, fill=True, thresh=0.05, cbar=True)
    plt.show()


def find_convertible_columns(df, print_output=False):
    """
    Find all columns in the DataFrame that can be converted to either integer or float data type.

    Parameters:
        df (DataFrame): DataFrame to search for convertible columns.
        print_output (bool): Boolean flag to control printing the transformed DataFrame and additional output.

    Returns:
        transformed_df (DataFrame): DataFrame with converted columns.
        convertible_columns (list): List of column names that were successfully converted.
        pending_transformation (list or False): List of column names that require further transformation,
                                                or False if there are none.
    """
    convertible_columns = []
    pending_transformation = []

    transformed_df = df.apply(lambda x: pd.to_numeric(x, errors='coerce') if pd.api.types.is_numeric_dtype(x) else x)

    for column in df.columns:
        if not pd.api.types.is_numeric_dtype(transformed_df[column]):
            pending_transformation.append(column)
        else:
            convertible_columns.append(column)

    if print_output:
        print("Transformed DataFrame:")
        print(transformed_df)
        print("\nData Types of Converted Columns:")
        print(transformed_df.dtypes[convertible_columns])
        print("\nData Types of Columns Pending Transformation:")
        print(df.dtypes[pending_transformation])
        print("\nColumns Pending Transformation:")
        print(pending_transformation)

    if pending_transformation:
        return transformed_df[convertible_columns], pending_transformation
    else:
        print('All columns are numeric')
        return False


def plots(df, x, y):
    """
    Generate KDE plots for the specified columns in the DataFrame.

    Parameters:
        df (DataFrame): DataFrame containing the data.
        x (str): Name of the column to plot on the x-axis.
        y (str): Name of the column to plot on the y-axis.
    """
    sns.kdeplot(df.cluster_labels_number_customer_service_calls_total_intl_charge,
                n_levels=20, fill=True, thresh=0.05, cbar=True)
    plt.show()
    sns.kdeplot(data=df, x=x, y=y, n_levels=20, fill=True, thresh=0.05, cbar=True)
    plt.show()

    result = find_convertible_columns(df)

    columns = ['cluster_labels_number_customer_service_calls_total_intl_charge', 'churn', x]

    fig, axes = plt.subplots(len(columns), 1, figsize=(10, 8), sharex=True)

    for i, column in enumerate(columns):
        sns.kdeplot(data=df, x=column, ax=axes[i])
        axes[i].set_title(column)

    plt.tight_layout()
    plt.show()


def kde_discretization(data, num_intervals=None):
    """
    Perform KDE-based discretization on a given dataset.

    Parameters:
        data (pandas.DataFrame): The input DataFrame containing continuous numerical data.
        num_intervals (int): The desired number of intervals for discretization. If None, it is set to the square of
                             the number of rows.

    Returns:
        discretized_data (array-like): The discretized data where each data point is assigned to a corresponding interval.
        intervals (list): The boundaries defining the intervals for discretization.

    """
    if num_intervals is None:
        num_intervals = int(np.sqrt(len(data))) ** 2

    data_values = data.values.flatten()

    kde = gaussian_kde(data_values)
    bandwidth = (1.06 * np.std(data_values) * len(data_values) ** (-0.2))
    min_value = np.min(data_values)
    max_value = np.max(data_values)
    points = np.linspace(min_value, max_value, num=1000)
    density_values = kde.evaluate(points)

    peaks = []
    for i in range(1, len(density_values) - 1):
        if density_values[i] > density_values[i - 1] and density_values[i] > density_values[i + 1]:
            peaks.append(points[i])
    peaks.sort()

    intervals = []
    intervals.append(min_value)
    if len(peaks) > 1:
        for i in range(len(peaks) - 1):
            intervals.append((peaks[i] + peaks[i + 1]) / 2)
    intervals.append(max_value)

    discretized_data = np.digitize(data_values, intervals) - 1

    return discretized_data, intervals


def eval_cluster(labels, data):
    """
    Evaluate the clustering results using various metrics.

    Parameters:
        labels (array-like): Cluster labels assigned to each data point.
        data (array-like): Data used for clustering.

    """
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

    silhouette_avg = silhouette_score(data, labels)
    print("Silhouette Score:", silhouette_avg)

    calinski_score = calinski_harabasz_score(data, labels)
    print("Calinski-Harabasz Index:", calinski_score)

    davies_bouldin_score = davies_bouldin_score(data, labels)
    print("Davies-Bouldin Index:", davies_bouldin_score)


def print_confusion_matrix(df):
    """
    Print the confusion matrix based on the cluster labels and true labels.

    Parameters:
        df (DataFrame): DataFrame containing the cluster labels and true labels.
    """
    confusion = confusion_matrix(df['churn'], df['cluster_labels_number_customer_service_calls_total_intl_charge'])
    sns.heatmap(confusion, annot=True, cmap='Blues')
    plt.xlabel('Clustered Variable')
    plt.ylabel('Target Variable')
    plt.show()

    evaluation_df = pd.DataFrame({'number_customer_service_calls': df['number_customer_service_calls'],
                                  'total_intl_charge': df['total_intl_charge'],
                                  'cluster_labels': df['cluster_labels_number_customer_service_calls_total_intl_charge']})

    confusion = confusion_matrix(evaluation_df['cluster_labels'], evaluation_df['number_customer_service_calls'])

    print("Confusion Matrix:")
    print(confusion)


def select_col_to_transform(df, dtype='int32', threshold=6):
    """
    Selects columns from the DataFrame for transformation based on the unique value threshold.

    Parameters:
        df (DataFrame): The input DataFrame.
        dtype (str): The data type of columns to consider for transformation. Default is 'int32'.
        threshold (int): The minimum number of unique values required for a column to be selected. Default is 6.

    Returns:
        list or None: A list of selected columns to transform. Returns None if no significant columns are found.
    """
    selected_columns = [column for column in df.select_dtypes(include=dtype) if np.unique(df[column]).size >= threshold]

    if len(selected_columns) == 0:
        print('No significant columns')
        return None

    return selected_columns


def main():
    df, x_train, x_test, y_train, y_test = load_dataset(desc=True)
    print(df.columns)
    y = 'total_intl_charge'
    x = 'number_customer_service_calls'
    df = add_cluster_labels(df, [x, y], 6)
    # plots(df, x, y)
    # print_confusion_matrix(df)
    selected_columns = [column for column in df.select_dtypes(include='int32')]
    print(df[selected_columns])
    while True:
        cols_to_transform = select_col_to_transform(df, 7)
        if len(cols_to_transform) == 0:
            break
        for column in cols_to_transform:
            data, num_intervals = kde_discretization(df[column])
            print(data, num_intervals)
        break


if __name__ == '__main__':
    main()
