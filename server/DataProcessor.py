class DataProcessor:
    def __init__(self):
        pass

    def one_hot_encode(self,data_set,arrExcludedColumns=[]):
        if arrExcludedColumns not in data_set:
            condition = data_set.columns
        else:
            condition = data_set.columns[~data_set.columns.isin(arrExcludedColumns)]
        for column in condition:
            if data_set[column].dtype not in ['float64', 'int64']:
                distinct_values = data_set[column].unique()  # Get distinct values
                mapping = {value: index for index, value in enumerate(distinct_values)}  # Create a mapping of values to indices
                print("Print mapping for column: ", column, " ||| ", mapping )
                for index, row in data_set.iterrows():
                    value = row[column]
                    if value in mapping:
                        data_set.at[index, column] = mapping[value]  # Replace value with corresponding index
        return data_set

    def process_csv(self,data_set):

        # Calculate the total number of calls made by a customer
        data_set['total_calls'] = data_set['total_day_calls'] + data_set['total_eve_calls'] + data_set['total_night_calls']

        # Calculate the total number of minutes used by a customer
        data_set['total_minutes'] = data_set['total_day_minutes'] + data_set['total_eve_minutes'] + data_set['total_night_minutes']

        # Calculate the ratio of international calls to total calls
        data_set['ratio_calls'] = data_set['total_intl_calls'] / data_set['total_calls']

        # Calculate the ratio of international minutes to total minutes
        data_set['ratio_minutes'] = data_set['total_intl_minutes'] / data_set['total_minutes']

        # Calculate the average number of minutes used during day calls per month
        data_set['avg_day_minutes'] = data_set['total_day_minutes'] / data_set['account_length']

        # Calculate the average number of minutes used during evening calls per month
        data_set['avg_eve_minutes'] = data_set['total_eve_minutes'] / data_set['account_length']

        # Calculate the average number of minutes used during night calls per month
        data_set['avg_night_minutes'] = data_set['total_night_minutes'] / data_set['account_length']

        # Calculate the total local charge incurred by a customer
        data_set['total_local_charge'] = data_set['total_day_charge'] + data_set['total_eve_charge'] + data_set['total_night_charge']

        # Calculate the ratio of international charge to total local charge
        data_set['ratio_charge'] = data_set['total_intl_charge'] / data_set['total_local_charge']

        return data_set
    
    def process_row(self,row):

        # Calculate the total number of calls made by a customer
        row['total_calls'] = row['total_day_calls'] + row['total_eve_calls'] + row['total_night_calls']

        # Calculate the total number of minutes used by a customer
        row['total_minutes'] = row['total_day_minutes'] + row['total_eve_minutes'] + row['total_night_minutes']

        # Calculate the ratio of international calls to total calls
        row['ratio_calls'] = row['total_intl_calls'] / row['total_calls']

        # Calculate the ratio of international minutes to total minutes
        row['ratio_minutes'] = row['total_intl_minutes'] / row['total_minutes']

        # Calculate the average number of minutes used during day calls per month
        row['avg_day_minutes'] = row['total_day_minutes'] / row['account_length']

        # Calculate the average number of minutes used during evening calls per month
        row['avg_eve_minutes'] = row['total_eve_minutes'] / row['account_length']

        # Calculate the average number of minutes used during night calls per month
        row['avg_night_minutes'] = row['total_night_minutes'] / row['account_length']

        # Calculate the total local charge incurred by a customer
        row['total_local_charge'] = row['total_day_charge'] + row['total_eve_charge'] + row['total_night_charge']

        # Calculate the ratio of international charge to total local charge
        row['ratio_charge'] = row['total_intl_charge'] / row['total_local_charge']

        return row