# Experiment for Eqiupment Failure using SVM or Random Forest
# Utilizing Nasa's Data which consist or IDK

#Analyze the data and find my own patterns and look for any correlations or columns that do not contribute

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dataset import load_dataset
from rfa_final import load_feats

def print_column_info(df):
    for i, col in enumerate(df.columns):
        print(f"{i} , {col}")
def heatmap_corr(df):
    heat = df.corr()
    plt.figure(figsize=[16,8])
    plt.title("Correlation between numerical features", size = len(df), pad = 20, color = '#8cabb6')
    sns.heatmap(heat,cmap = sns.diverging_palette(20, 220, n = 200), annot=False)
    plt.show()

def bar_corr_plot(df):
    plt.figure(figsize=(15,8))
    df.corr()['churn'].sort_values(ascending = False).plot(kind='bar', rot=180)
    plt.show()

def added_features(df):

    # State Categories
    df['state_category'] = pd.cut(df['state'], bins=[0, 10, 20, 30, 40], labels=['A', 'B', 'C', 'D'])

    # Account Length Categories
    df['account_length_category'] = pd.cut(df['account_length'], bins=[0, 100, 200, 300, 400, np.inf], labels=['short-term', 'medium-term', 'long-term', 'extra-long-term', 'unknown'])

    # Area Code Categories
    df['area_code_category'] = df['area_code'].astype(str).apply(lambda x: 'Region A' if x.startswith('A') else 'Region B' if x.startswith('B') else 'Region C')

    # International Plan and Voice Mail Plan Indicator
    df['international_plan_indicator'] = df['international_plan'].map({'no': 0, 'yes': 1})
    df['voice_mail_plan_indicator'] = df['voice_mail_plan'].map({'no': 0, 'yes': 1})

    # Total Voice Mail Messages Indicator
    df['has_voice_mail_messages'] = (df['number_vmail_messages'] > 0).astype(int)

    # Total Call Charges
    df['total_call_charges'] = df['total_day_charge'] + df['total_eve_charge'] + df['total_night_charge'] + df['total_intl_charge']

    # Total Call Minutes
    df['total_call_minutes'] = df['total_day_minutes'] + df['total_eve_minutes'] + df['total_night_minutes'] + df['total_intl_minutes']

    # Average Call Duration
    df['average_call_duration'] = df['total_call_minutes'] / (df['total_day_calls'] + df['total_eve_calls'] + df['total_night_calls'] + df['total_intl_calls'])

    # Call Density
    df['call_density_day'] = df['total_day_calls'] / df['total_day_minutes']
    df['call_density_evening'] = df['total_eve_calls'] / df['total_eve_minutes']
    df['call_density_night'] = df['total_night_calls'] / df['total_night_minutes']
    df['call_density_intl'] = df['total_intl_calls'] / df['total_intl_minutes']

    # Service Calls Categories
    df['service_calls_category'] = pd.cut(df['number_customer_service_calls'], bins=[0, 2, 4, np.inf], labels=['low', 'medium', 'high'])

    # Total Evening Call Charges
    df['total_evening_call_charges'] = df['total_eve_charge'] * df['total_eve_calls']

    # Total Night Call Charges
    df['total_night_call_charges'] = df['total_night_charge'] * df['total_night_calls']

    # Total International Call Charges
    #df['total_intl_call_charges'] = df['total_intl_charge'] * df['total_intl_calls']

    # Total Day/Night Call Ratio
    df['day_night_call_ratio'] = df['total_day_calls'] / df['total_night_calls']

    # Total Call Charges per Minute
    df['call_charges_per_minute'] = df['total_call_charges'] / df['total_call_minutes']

    # Average Call Charges per Service Call
    df['avg_charges_per_service_call'] = df['total_call_charges'] / df['number_customer_service_calls']

    # Call Duration per Customer Service Call
    df['call_duration_per_service_call'] = df['total_call_minutes'] / df['number_customer_service_calls']

    # Call Density per Area Code
    df['call_density_area_code'] = df['total_day_calls'] / df.groupby('area_code')['total_day_minutes'].transform('sum')

    # Call Density per State
    df['call_density_state'] = df['total_day_calls'] / df.groupby('state')['total_day_minutes'].transform('sum')

    # Total Calls per Account Length
    df['total_calls_per_account_length'] = df['total_day_calls'] + df['total_eve_calls'] + df['total_night_calls'] + df['total_intl_calls'] / df['account_length']

    # Total Charges per Account Length
    df['total_charges_per_account_length'] = df['total_day_charge'] + df['total_eve_charge'] + df['total_night_charge'] + df['total_intl_charge'] / df['account_length']

    # Average Call Charges per State
    df['average_charges_per_state'] = df.groupby('state')['total_call_charges'].transform('mean')

    # Average Call Minutes per State
    df['average_minutes_per_state'] = df.groupby('state')['total_call_minutes'].transform('mean')

    # Average Call Charges per Area Code
    df['average_charges_per_area_code'] = df.groupby('area_code')['total_call_charges'].transform('mean')

    # Average Call Minutes per Area Code
    df['average_minutes_per_area_code'] = df.groupby('area_code')['total_call_minutes'].transform('mean')

    # Call Duration per Number of Voice Mail Messages
    df['call_duration_per_vmail_messages'] = df['total_call_minutes'] / df['number_vmail_messages']

    # Total Calls per Customer Service Calls
    df['total_calls_per_service_calls'] = df['total_day_calls'] + df['total_eve_calls'] + df['total_night_calls'] + df['total_intl_calls'] / df['number_customer_service_calls']

    # Total Charges per Customer Service Calls
    df['total_charges_per_service_calls'] = df['total_day_charge'] + df['total_eve_charge'] + df['total_night_charge'] + df['total_intl_charge'] / df['number_customer_service_calls']

    # Ratio of Day Calls to Evening Calls
    df['day_to_evening_call_ratio'] = df['total_day_calls'] / df['total_eve_calls']

    # Ratio of Day Minutes to Evening Minutes
    df['day_to_evening_minutes_ratio'] = df['total_day_minutes'] / df['total_eve_minutes']

    # Ratio of Night Calls to International Calls
    df['night_to_intl_call_ratio'] = df['total_night_calls'] / df['total_intl_calls']

    # Ratio of Night Minutes to International Minutes
    df['night_to_intl_minutes_ratio'] = df['total_night_minutes'] / df['total_intl_minutes']

    # Average Call Charges per Number of Voice Mail Messages
    df['avg_charges_per_vmail_message'] = df['total_call_charges'] / df['number_vmail_messages']

    # Average Call Minutes per Number of Voice Mail Messages
    df['avg_minutes_per_vmail_message'] = df['total_call_minutes'] / df['number_vmail_messages']

    # Average Call Charges per Customer Service Calls
    df['avg_charges_per_service_calls'] = df['total_call_charges'] / df['number_customer_service_calls']

    # Average Call Minutes per Customer Service Calls
    df['avg_minutes_per_service_calls'] = df['total_call_minutes'] / df['number_customer_service_calls']

    # Number of Total Calls minus Number of Voice Mail Messages
    df['calls_minus_vmail_messages'] = df['total_day_calls'] + df['total_eve_calls'] + df['total_night_calls'] + df['total_intl_calls'] - df['number_vmail_messages']

    # Total Charges divided by Total Call Minutes
    df['charges_per_minute'] = df['total_call_charges'] / df['total_call_minutes']

    # Average Call Charges per State and Area Code combination
    df['avg_charges_per_state_area'] = df.groupby(['state', 'area_code'])['total_call_charges'].transform('mean')

    # Average Call Minutes per State and Area Code combination
    df['avg_minutes_per_state_area'] = df.groupby(['state', 'area_code'])['total_call_minutes'].transform('mean')

    # Average Call Charges per Account Length
    df['average_charges_per_account_length'] = df['total_call_charges'] / df['account_length']

    # Average Call Minutes per Account Length
    df['average_minutes_per_account_length'] = df['total_call_minutes'] / df['account_length']

    # Total Calls per International Plan
    df['total_calls_per_international_plan'] = df['total_day_calls'] + df['total_eve_calls'] + df['total_night_calls'] + df['total_intl_calls'] * df['international_plan_indicator']

    # Total Charges per International Plan
    df['total_charges_per_international_plan'] = df['total_day_charge'] + df['total_eve_charge'] + df['total_night_charge'] + df['total_intl_charge'] * df['international_plan_indicator']

    # Total Calls per Voice Mail Plan
    df['total_calls_per_voice_mail_plan'] = df['total_day_calls'] + df['total_eve_calls'] + df['total_night_calls'] + df['total_intl_calls'] * df['voice_mail_plan_indicator']

    # Total Charges per Voice Mail Plan
    df['total_charges_per_voice_mail_plan'] = df['total_day_charge'] + df['total_eve_charge'] + df['total_night_charge'] + df['total_intl_charge'] * df['voice_mail_plan_indicator']

    # Total Minutes per Number of Customer Service Calls
    df['total_minutes_per_service_calls'] = df['total_day_minutes'] + df['total_eve_minutes'] + df['total_night_minutes'] + df['total_intl_minutes'] / df['number_customer_service_calls']

    # Total Charges per Number of Customer Service Calls
    df['total_charges_per_service_calls'] = df['total_day_charge'] + df['total_eve_charge'] + df['total_night_charge'] + df['total_intl_charge'] / df['number_customer_service_calls']

    # Ratio of Day Charges to Total Charges
    df['day_charge_ratio'] = df['total_day_charge'] / (df['total_day_charge'] + df['total_eve_charge'] + df['total_night_charge'] + df['total_intl_charge'])

    # Ratio of Evening Charges to Total Charges
    df['evening_charge_ratio'] = df['total_eve_charge'] / (df['total_day_charge'] + df['total_eve_charge'] + df['total_night_charge'] + df['total_intl_charge'])

    # Ratio of Night Charges to Total Charges
    df['night_charge_ratio'] = df['total_night_charge'] / (df['total_day_charge'] + df['total_eve_charge'] + df['total_night_charge'] + df['total_intl_charge'])

    # Ratio of International Charges to Total Charges
    df['intl_charge_ratio'] = df['total_intl_charge'] / (df['total_day_charge'] + df['total_eve_charge'] + df['total_night_charge'] + df['total_intl_charge'])          
    # Total Call Charges per Day Calls
    df['charges_per_day_calls'] = df['total_day_charge'] / df['total_day_calls']

    # Total Call Charges per Evening Calls
    df['charges_per_evening_calls'] = df['total_eve_charge'] / df['total_eve_calls']

    # Total Call Charges per Night Calls
    df['charges_per_night_calls'] = df['total_night_charge'] / df['total_night_calls']

    # Total Call Charges per International Calls
    df['charges_per_intl_calls'] = df['total_intl_charge'] / df['total_intl_calls']

    # Average Call Duration per Day Calls
    df['avg_duration_per_day_calls'] = df['total_day_minutes'] / df['total_day_calls']

    # Average Call Duration per Evening Calls
    df['avg_duration_per_evening_calls'] = df['total_eve_minutes'] / df['total_eve_calls']

    # Average Call Duration per Night Calls
    df['avg_duration_per_night_calls'] = df['total_night_minutes'] / df['total_night_calls']

    # Average Call Duration per International Calls
    df['avg_duration_per_intl_calls'] = df['total_intl_minutes'] / df['total_intl_calls']

    # Percentage of Day Calls in Total Calls
    df['percentage_day_calls'] = (df['total_day_calls'] / (df['total_day_calls'] + df['total_eve_calls'] + df['total_night_calls'] + df['total_intl_calls'])) * 100

    # Percentage of Evening Calls in Total Calls
    df['percentage_evening_calls'] = (df['total_eve_calls'] / (df['total_day_calls'] + df['total_eve_calls'] + df['total_night_calls'] + df['total_intl_calls'])) * 100

    # Percentage of Night Calls in Total Calls
    df['percentage_night_calls'] = (df['total_night_calls'] / (df['total_day_calls'] + df['total_eve_calls'] + df['total_night_calls'] + df['total_intl_calls'])) * 100

    # Percentage of International Calls in Total Calls
    df['percentage_intl_calls'] = (df['total_intl_calls'] / (df['total_day_calls'] + df['total_eve_calls'] + df['total_night_calls'] + df['total_intl_calls'])) * 100

    # Total Call Minutes per Number of Customer Service Calls
    df['total_minutes_per_service_calls'] = df['total_call_minutes'] / df['number_customer_service_calls']

    # Total Call Charges per Number of Customer Service Calls
    df['total_charges_per_service_calls'] = df['total_call_charges'] / df['number_customer_service_calls']

    # Average Call Minutes per Number of Voice Mail Messages
    df['avg_minutes_per_vmail_messages'] = df['total_call_minutes'] / df['number_vmail_messages']

    # Average Call Charges per Number of Voice Mail Messages
    df['avg_charges_per_vmail_messages'] = df['total_call_charges'] / df['number_vmail_messages']

    # Total Minutes per Total Calls
    df['minutes_per_total_calls'] = df['total_call_minutes'] / (df['total_day_calls'] + df['total_eve_calls'] + df['total_night_calls'] + df['total_intl_calls'])

    # Total Charges per Total Calls
    df['charges_per_total_calls'] = df['total_call_charges'] / (df['total_day_calls'] + df['total_eve_calls'] + df['total_night_calls'] + df['total_intl_calls'])

    # Ratio of Evening Calls to Total Calls
    df['evening_calls_ratio'] = df['total_eve_calls'] / (df['total_day_calls'] + df['total_eve_calls'] + df['total_night_calls'] + df['total_intl_calls'])

    # Ratio of Night Calls to Total Calls
    df['night_calls_ratio'] = df['total_night_calls'] / (df['total_day_calls'] + df['total_eve_calls'] + df['total_night_calls'] + df['total_intl_calls'])

    # Ratio of International Calls to Total Calls
    df['intl_calls_ratio'] = df['total_intl_calls'] / (df['total_day_calls'] + df['total_eve_calls'] + df['total_night_calls'] + df['total_intl_calls'])

    # Average Call Minutes per Customer Service Calls
    df['avg_minutes_per_service_calls'] = df['total_call_minutes'] / df['number_customer_service_calls']

    # Average Call Charges per Customer Service Calls
    df['avg_charges_per_service_calls'] = df['total_call_charges'] / df['number_customer_service_calls']

    # Percentage of Customers with International Plan
    #df['percentage_international_plan'] = (df['international_plan'] == 'yes').astype(int) * 100

    # Percentage of Customers with Voice Mail Plan
    df['percentage_voice_mail_plan'] = (df['voice_mail_plan'] == 'yes').astype(int) * 100

    # Total Charges per Total Minutes
    df['charges_per_total_minutes'] = df['total_call_charges'] / (df['total_call_minutes'] + 1)  # Adding 1 to avoid division by zero

    # Average Call Charges per State
    df['average_charges_per_state'] = df.groupby('state')['total_call_charges'].transform('mean')

    # Average Call Minutes per State
    df['average_minutes_per_state'] = df.groupby('state')['total_call_minutes'].transform('mean')

    # Average Call Charges per Area Code
    df['average_charges_per_area_code'] = df.groupby('area_code')['total_call_charges'].transform('mean')

    # Average Call Minutes per Area Code
    df['average_minutes_per_area_code'] = df.groupby('area_code')['total_call_minutes'].transform('mean')

    # Percentage of Customers with International Plan by State
    df['percentage_international_plan_by_state'] = df.groupby('state')['international_plan_indicator'].transform('mean') * 100

    # Percentage of Customers with Voice Mail Plan by State
    df['percentage_voice_mail_plan_by_state'] = df.groupby('state')['voice_mail_plan_indicator'].transform('mean') * 100

    # Total Call Charges per Total Call Minutes
    df['charges_per_minutes'] = df['total_call_charges'] / df['total_call_minutes']

    # Total Call Minutes per Total Call Charges
    df['minutes_per_charges'] = df['total_call_minutes'] / df['total_call_charges']

    # Average Call Charges per Number of Night Calls
    df['average_charges_per_night_calls'] = df['total_night_charge'] / df['total_night_calls']

    # Average Call Minutes per Number of Night Calls
    df['average_minutes_per_night_calls'] = df['total_night_minutes'] / df['total_night_calls']

    # Average Call Charges per Number of International Calls
    df['average_charges_per_intl_calls'] = df['total_intl_charge'] / df['total_intl_calls']

    # Average Call Minutes per Number of International Calls
    df['average_minutes_per_intl_calls'] = df['total_intl_minutes'] / df['total_intl_calls']

    # Ratio of Voice Mail Messages to Total Call Minutes
    df['vmail_messages_to_minutes_ratio'] = df['number_vmail_messages'] / df['total_call_minutes']

    # Ratio of Customer Service Calls to Total Call Minutes
    df['service_calls_to_minutes_ratio'] = df['number_customer_service_calls'] / df['total_call_minutes']

    # Difference between Total Day Minutes and Total Evening Minutes
    df['day_evening_minutes_difference'] = df['total_day_minutes'] - df['total_eve_minutes']

    # Difference between Total Night Minutes and Total International Minutes
    df['night_intl_minutes_difference'] = df['total_night_minutes'] - df['total_intl_minutes']
    # Total Call Minutes per Total Call Charges per State
    df['minutes_per_charges_per_state'] = df['total_call_minutes'] / (df.groupby('state')['total_call_charges'].transform('sum'))

    # Total Charges per Total Call Minutes per State
    df['charges_per_minutes_per_state'] = df['total_call_charges'] / (df.groupby('state')['total_call_minutes'].transform('sum'))

    # Percentage of Customers with International Plan per State
    df['percentage_international_plan_per_state'] = (df.groupby('state')['international_plan_indicator'].transform('mean')) * 100

    # Percentage of Customers with Voice Mail Plan per State
    df['percentage_voice_mail_plan_per_state'] = (df.groupby('state')['voice_mail_plan_indicator'].transform('mean')) * 100

    # Total Call Charges per Total Call Minutes per Area Code
    df['charges_per_minutes_per_area_code'] = df['total_call_charges'] / (df.groupby('area_code')['total_call_minutes'].transform('sum'))

    # Total Call Minutes per Total Call Charges per Area Code
    df['minutes_per_charges_per_area_code'] = df['total_call_minutes'] / (df.groupby('area_code')['total_call_charges'].transform('sum'))

    # Difference between Total Day Minutes and Total Evening Minutes per State
    df['day_evening_minutes_difference_per_state'] = df.groupby('state')['total_day_minutes'].transform('mean') - df.groupby('state')['total_eve_minutes'].transform('mean')

    # Difference between Total Night Minutes and Total International Minutes per State
    df['night_intl_minutes_difference_per_state'] = df.groupby('state')['total_night_minutes'].transform('mean') - df.groupby('state')['total_intl_minutes'].transform('mean')

    # Total Call Charges per Total Call Minutes per International Plan
    df['charges_per_minutes_per_international_plan'] = df['total_call_charges'] / (df.groupby('international_plan')['total_call_minutes'].transform('sum'))

    # Total Call Minutes per Total Call Charges per International Plan
    df['minutes_per_charges_per_international_plan'] = df['total_call_minutes'] / (df.groupby('international_plan')['total_call_charges'].transform('sum'))

    # Average Call Minutes per Number of Customer Service Calls per State
    df['avg_minutes_per_service_calls_per_state'] = df['total_call_minutes'] / (df.groupby(['state', 'number_customer_service_calls'])['number_customer_service_calls'].transform('count'))

    # Average Call Charges per Number of Customer Service Calls per State
    df['avg_charges_per_service_calls_per_state'] = df['total_call_charges'] / (df.groupby(['state', 'number_customer_service_calls'])['number_customer_service_calls'].transform('count'))

    # Total Call Charges per Total Call Minutes per State and Area Code
    df['charges_per_minutes_per_state_area_code'] = df['total_call_charges'] / (df.groupby(['state', 'area_code'])['total_call_minutes'].transform('sum'))

    # Total Call Minutes per Total Call Charges per State and Area Code
    df['minutes_per_charges_per_state_area_code'] = df['total_call_minutes'] / (df.groupby(['state', 'area_code'])['total_call_charges'].transform('sum'))

    # Total Call Charges per Total Call Minutes per International Plan and Voice Mail Plan
    df['charges_per_minutes_per_plan'] = df['total_call_charges'] / (df.groupby(['international_plan', 'voice_mail_plan'])['total_call_minutes'].transform('sum'))

    # Total Call Minutes per Total Call Charges per International Plan and Voice Mail Plan
    df['minutes_per_charges_per_plan'] = df['total_call_minutes'] / (df.groupby(['international_plan', 'voice_mail_plan'])['total_call_charges'].transform('sum'))

    # Difference between Total Day Minutes and Total Evening Minutes per Area Code
    df['day_evening_minutes_difference_per_area_code'] = df.groupby('area_code')['total_day_minutes'].transform('mean') - df.groupby('area_code')['total_eve_minutes'].transform('mean')

    # Difference between Total Night Minutes and Total International Minutes per Area Code
    df['night_intl_minutes_difference_per_area_code'] = df.groupby('area_code')['total_night_minutes'].transform('mean') - df.groupby('area_code')['total_intl_minutes'].transform('mean')

    # Average Call Charges per Number of Voice Mail Messages per State
    df['avg_charges_per_vmail_messages_per_state'] = df['total_call_charges'] / (df.groupby(['state', 'number_vmail_messages'])['number_vmail_messages'].transform('count'))

    # Average Call Minutes per Number of Voice Mail Messages per State
    df['avg_minutes_per_vmail_messages_per_state'] = df['total_call_minutes'] / (df.groupby(['state', 'number_vmail_messages'])['number_vmail_messages'].transform('count'))
        
    # Total Call Minutes per Total Call Charges per International Plan and State
    df['minutes_per_charges_per_intl_plan_state'] = df['total_call_minutes'] / (df.groupby(['international_plan', 'state'])['total_call_charges'].transform('sum'))

    # Total Call Charges per Total Call Minutes per International Plan and State
    df['charges_per_minutes_per_intl_plan_state'] = df['total_call_charges'] / (df.groupby(['international_plan', 'state'])['total_call_minutes'].transform('sum'))

    # Ratio of Total Evening Minutes to Total Day Minutes
    df['eve_day_minutes_ratio'] = df['total_eve_minutes'] / df['total_day_minutes']

    # Ratio of Total Night Minutes to Total Day Minutes
    df['night_day_minutes_ratio'] = df['total_night_minutes'] / df['total_day_minutes']

    # Ratio of Total International Minutes to Total Day Minutes
    df['intl_day_minutes_ratio'] = df['total_intl_minutes'] / df['total_day_minutes']

    # Ratio of Total Evening Calls to Total Day Calls
    df['eve_day_calls_ratio'] = df['total_eve_calls'] / df['total_day_calls']

    # Ratio of Total Night Calls to Total Day Calls
    df['night_day_calls_ratio'] = df['total_night_calls'] / df['total_day_calls']

    # Ratio of Total International Calls to Total Day Calls
    df['intl_day_calls_ratio'] = df['total_intl_calls'] / df['total_day_calls']

    # Difference between Total Evening Calls and Total Day Calls
    df['eve_day_calls_difference'] = df['total_eve_calls'] - df['total_day_calls']

    # Difference between Total Night Calls and Total Day Calls
    df['night_day_calls_difference'] = df['total_night_calls'] - df['total_day_calls']

    # Difference between Total International Calls and Total Day Calls
    df['intl_day_calls_difference'] = df['total_intl_calls'] - df['total_day_calls']

    # Ratio of Total Evening Charges to Total Day Charges
    df['eve_day_charges_ratio'] = df['total_eve_charge'] / df['total_day_charge']

    # Ratio of Total Night Charges to Total Day Charges
    df['night_day_charges_ratio'] = df['total_night_charge'] / df['total_day_charge']

    # Ratio of Total International Charges to Total Day Charges
    df['intl_day_charges_ratio'] = df['total_intl_charge'] / df['total_day_charge']

    # Difference between Total Evening Charges and Total Day Charges
    df['eve_day_charges_difference'] = df['total_eve_charge'] - df['total_day_charge']

    # Difference between Total Night Charges and Total Day Charges
    df['night_day_charges_difference'] = df['total_night_charge'] - df['total_day_charge']

    # Difference between Total International Charges and Total Day Charges
    df['intl_day_charges_difference'] = df['total_intl_charge'] - df['total_day_charge']

    # Total Call Minutes per Total Call Charges per Voice Mail Plan
    df['minutes_per_charges_per_vmail_plan'] = df['total_call_minutes'] / (df.groupby('voice_mail_plan')['total_call_charges'].transform('sum'))

    # Total Call Charges per Total Call Minutes per Voice Mail Plan
    df['charges_per_minutes_per_vmail_plan'] = df['total_call_charges'] / (df.groupby('voice_mail_plan')['total_call_minutes'].transform('sum'))

    # Total Call Minutes per Total Call Charges per State and Voice Mail Plan
    df['minutes_per_charges_per_state_vmail_plan'] = df['total_call_minutes'] / (df.groupby(['state', 'voice_mail_plan'])['total_call_charges'].transform('sum'))

    # Total Call Charges per Total Call Minutes per State and Voice Mail Plan
    df['charges_per_minutes_per_state_vmail_plan'] = df['total_call_charges'] / (df.groupby(['state', 'voice_mail_plan'])['total_call_minutes'].transform('sum'))

    # Difference between Total Day Minutes and Total Night Minutes per State
    df['day_night_minutes_difference_per_state'] = df.groupby('state')['total_day_minutes'].transform('mean') - df.groupby('state')['total_night_minutes'].transform('mean')

    # Difference between Total Evening Minutes and Total International Minutes per State
    df['eve_intl_minutes_difference_per_state'] = df.groupby('state')['total_eve_minutes'].transform('mean') - df.groupby('state')['total_intl_minutes'].transform('mean')

    # Average Call Charges per Number of Customer Service Calls
    df['avg_charges_per_service_calls'] = df['total_call_charges'] / df['number_customer_service_calls']

    # Average Call Minutes per Number of Customer Service Calls
    df['avg_minutes_per_service_calls'] = df['total_call_minutes'] / df['number_customer_service_calls']

    # Ratio of Total Evening Calls to Total Night Calls
    df['eve_night_calls_ratio'] = df['total_eve_calls'] / df['total_night_calls']

    # Ratio of Total International Calls to Total Night Calls
    df['intl_night_calls_ratio'] = df['total_intl_calls'] / df['total_night_calls']

    # Ratio of Total Evening Charges to Total Night Charges
    df['eve_night_charges_ratio'] = df['total_eve_charge'] / df['total_night_charge']

    # Ratio of Total International Charges to Total Night Charges
    df['intl_night_charges_ratio'] = df['total_intl_charge'] / df['total_night_charge']

    # Difference between Total Evening Calls and Total Night Calls
    df['eve_night_calls_difference'] = df['total_eve_calls'] - df['total_night_calls']

    # Difference between Total International Calls and Total Night Calls
    df['intl_night_calls_difference'] = df['total_intl_calls'] - df['total_night_calls']

    # Difference between Total Evening Charges and Total Night Charges
    df['eve_night_charges_difference'] = df['total_eve_charge'] - df['total_night_charge']

    # Difference between Total International Charges and Total Night Charges
    df['intl_night_charges_difference'] = df['total_intl_charge'] - df['total_night_charge']

    # Average Call Minutes per Number of Voice Mail Messages
    df['avg_minutes_per_vmail_messages'] = df['total_call_minutes'] / df['number_vmail_messages']

    # Average Call Charges per Number of Voice Mail Messages
    df['avg_charges_per_vmail_messages'] = df['total_call_charges'] / df['number_vmail_messages']

    # Ratio of Total Evening Minutes to Total International Minutes
    df['eve_intl_minutes_ratio'] = df['total_eve_minutes'] / df['total_intl_minutes']

    # Ratio of Total Evening Calls to Total International Calls
    df['eve_intl_calls_ratio'] = df['total_eve_calls'] / df['total_intl_calls']

    # Ratio of Total Evening Charges to Total International Charges
    df['eve_intl_charges_ratio'] = df['total_eve_charge'] / df['total_intl_charge']

    # Difference between Total Evening Minutes and Total Day Minutes per Area Code
    df['eve_day_minutes_difference_per_area_code'] = df.groupby('area_code')['total_eve_minutes'].transform('mean') - df.groupby('area_code')['total_day_minutes'].transform('mean')

    # Difference between Total Evening Calls and Total Day Calls per Area Code
    df['eve_day_calls_difference_per_area_code'] = df.groupby('area_code')['total_eve_calls'].transform('mean') - df.groupby('area_code')['total_day_calls'].transform('mean')

    # Difference between Total Evening Charges and Total Day Charges per Area Code
    df['eve_day_charges_difference_per_area_code'] = df.groupby('area_code')['total_eve_charge'].transform('mean') - df.groupby('area_code')['total_day_charge'].transform('mean')

    # Average Call Minutes per Number of International Calls
    df['avg_minutes_per_intl_calls'] = df['total_call_minutes'] / df['total_intl_calls']

    # Average Call Charges per Number of International Calls
    df['avg_charges_per_intl_calls'] = df['total_call_charges'] / df['total_intl_calls']

    # Ratio of Total Night Minutes to Total International Minutes
    df['night_intl_minutes_ratio'] = df['total_night_minutes'] / df['total_intl_minutes']

    # Ratio of Total Night Calls to Total International Calls
    df['night_intl_calls_ratio'] = df['total_night_calls'] / df['total_intl_calls']

    # Ratio of Total Night Charges to Total International Charges
    df['night_intl_charges_ratio'] = df['total_night_charge'] / df['total_intl_charge']

    # Difference between Total Night Minutes and Total Day Minutes per International Plan
    df['night_day_minutes_difference_per_intl_plan'] = df.groupby('international_plan')['total_night_minutes'].transform('mean') - df.groupby('international_plan')['total_day_minutes'].transform('mean')

    # Difference between Total Night Calls and Total Day Calls per International Plan
    df['night_day_calls_difference_per_intl_plan'] = df.groupby('international_plan')['total_night_calls'].transform('mean') - df.groupby('international_plan')['total_day_calls'].transform('mean')

    # Difference between Total Night Charges and Total Day Charges per International Plan
    df['night_day_charges_difference_per_intl_plan'] = df.groupby('international_plan')['total_night_charge'].transform('mean') - df.groupby('international_plan')['total_day_charge'].transform('mean')
    
    # Average Call Minutes per Number of Total International Minutes
    df['avg_minutes_per_total_intl_minutes'] = df['total_call_minutes'] / df['total_intl_minutes']

    # Average Call Charges per Number of Total International Minutes
    df['avg_charges_per_total_intl_minutes'] = df['total_call_charges'] / df['total_intl_minutes']

    # Ratio of Total Day Calls to Total International Calls
    df['day_intl_calls_ratio'] = df['total_day_calls'] / df['total_intl_calls']

    # Ratio of Total Day Charges to Total International Charges
    df['day_intl_charges_ratio'] = df['total_day_charge'] / df['total_intl_charge']

    # Difference between Total Day Minutes and Total Night Minutes per State
    df['day_night_minutes_difference_per_state'] = df.groupby('state')['total_day_minutes'].transform('mean') - df.groupby('state')['total_night_minutes'].transform('mean')

    # Difference between Total Day Calls and Total Night Calls per State
    df['day_night_calls_difference_per_state'] = df.groupby('state')['total_day_calls'].transform('mean') - df.groupby('state')['total_night_calls'].transform('mean')

    # Difference between Total Day Charges and Total Night Charges per State
    df['day_night_charges_difference_per_state'] = df.groupby('state')['total_day_charge'].transform('mean') - df.groupby('state')['total_night_charge'].transform('mean')

    # Ratio of Total Evening Minutes to Total Night Minutes per Area Code
    df['eve_night_minutes_ratio_per_area_code'] = df.groupby('area_code')['total_eve_minutes'].transform('mean') / df.groupby('area_code')['total_night_minutes'].transform('mean')

    # Ratio of Total Evening Calls to Total Night Calls per Area Code
    df['eve_night_calls_ratio_per_area_code'] = df.groupby('area_code')['total_eve_calls'].transform('mean') / df.groupby('area_code')['total_night_calls'].transform('mean')

    # Ratio of Total Evening Charges to Total Night Charges per Area Code
    df['eve_night_charges_ratio_per_area_code'] = df.groupby('area_code')['total_eve_charge'].transform('mean') / df.groupby('area_code')['total_night_charge'].transform('mean')

    # Difference between Total Evening Minutes and Total International Minutes per Voice Mail Plan
    df['eve_intl_minutes_difference_per_vmail_plan'] = df.groupby('voice_mail_plan')['total_eve_minutes'].transform('mean') - df.groupby('voice_mail_plan')['total_intl_minutes'].transform('mean')

    # Difference between Total Evening Calls and Total International Calls per Voice Mail Plan
    df['eve_intl_calls_difference_per_vmail_plan'] = df.groupby('voice_mail_plan')['total_eve_calls'].transform('mean') - df.groupby('voice_mail_plan')['total_intl_calls'].transform('mean')

    # Difference between Total Evening Charges and Total International Charges per Voice Mail Plan
    df['eve_intl_charges_difference_per_vmail_plan'] = df.groupby('voice_mail_plan')['total_eve_charge'].transform('mean') - df.groupby('voice_mail_plan')['total_intl_charge'].transform('mean')

    # Average Call Minutes per Number of Total Day Calls
    df['avg_minutes_per_total_day_calls'] = df['total_call_minutes'] / df['total_day_calls']

    # Average Call Charges per Number of Total Day Calls
    df['avg_charges_per_total_day_calls'] = df['total_call_charges'] / df['total_day_calls']

    # Ratio of Total Day Minutes to Total Night Minutes per International Plan
    df['day_night_minutes_ratio_per_intl_plan'] = df.groupby('international_plan')['total_day_minutes'].transform('mean') / df.groupby('international_plan')['total_night_minutes'].transform('mean')

    # Ratio of Total Day Calls to Total Night Calls per International Plan
    df['day_night_calls_ratio_per_intl_plan'] = df.groupby('international_plan')['total_day_calls'].transform('mean') / df.groupby('international_plan')['total_night_calls'].transform('mean')

    # Ratio of Total Day Charges to Total Night Charges per International Plan
    df['day_night_charges_ratio_per_intl_plan'] = df.groupby('international_plan')['total_day_charge'].transform('mean') / df.groupby('international_plan')['total_night_charge'].transform('mean')

    # Difference between Total Evening Minutes and Total Night Minutes per International Plan
    df['eve_night_minutes_difference_per_intl_plan'] = df.groupby('international_plan')['total_eve_minutes'].transform('mean') - df.groupby('international_plan')['total_night_minutes'].transform('mean')

    # Difference between Total Evening Calls and Total Night Calls per International Plan
    df['eve_night_calls_difference_per_intl_plan'] = df.groupby('international_plan')['total_eve_calls'].transform('mean') - df.groupby('international_plan')['total_night_calls'].transform('mean')

    # Difference between Total Evening Charges and Total Night Charges per International Plan
    df['eve_night_charges_difference_per_intl_plan'] = df.groupby('international_plan')['total_eve_charge'].transform('mean') - df.groupby('international_plan')['total_night_charge'].transform('mean')

    # Ratio of Total Evening Minutes to Total International Minutes per State and International Plan
    df['eve_intl_minutes_ratio_per_state_intl_plan'] = df.groupby(['state', 'international_plan'])['total_eve_minutes'].transform('mean') / df.groupby(['state', 'international_plan'])['total_intl_minutes'].transform('mean')

    # Ratio of Total Evening Calls to Total International Calls per State and International Plan
    df['eve_intl_calls_ratio_per_state_intl_plan'] = df.groupby(['state', 'international_plan'])['total_eve_calls'].transform('mean') / df.groupby(['state', 'international_plan'])['total_intl_calls'].transform('mean')

    # Ratio of Total Evening Charges to Total International Charges per State and International Plan
    df['eve_intl_charges_ratio_per_state_intl_plan'] = df.groupby(['state', 'international_plan'])['total_eve_charge'].transform('mean') / df.groupby(['state', 'international_plan'])['total_intl_charge'].transform('mean')

    # Ratio of Total Day Minutes to Total Day Calls
    df['day_minutes_calls_ratio'] = df['total_day_minutes'] / df['total_day_calls']

    # Ratio of Total Evening Minutes to Total Evening Calls
    df['eve_minutes_calls_ratio'] = df['total_eve_minutes'] / df['total_eve_calls']

    # Ratio of Total Night Minutes to Total Night Calls
    df['night_minutes_calls_ratio'] = df['total_night_minutes'] / df['total_night_calls']

    # Ratio of Total International Minutes to Total International Calls
    df['intl_minutes_calls_ratio'] = df['total_intl_minutes'] / df['total_intl_calls']

    # Average Call Duration per State
    df['avg_call_duration_per_state'] = df['total_call_minutes'] / df.groupby('state')['number_customer_service_calls'].transform('sum')

    # Average Call Duration per Area Code
    df['avg_call_duration_per_area_code'] = df['total_call_minutes'] / df.groupby('area_code')['number_customer_service_calls'].transform('sum')

    # Difference between Total Evening Minutes and Total Night Minutes
    df['eve_night_minutes_difference'] = df['total_eve_minutes'] - df['total_night_minutes']

    # Difference between Total Evening Calls and Total Night Calls
    df['eve_night_calls_difference'] = df['total_eve_calls'] - df['total_night_calls']

    # Difference between Total Evening Charges and Total Night Charges
    df['eve_night_charges_difference'] = df['total_eve_charge'] - df['total_night_charge']

    # Difference between Total International Minutes and Total International Calls
    df['intl_minutes_calls_difference'] = df['total_intl_minutes'] - df['total_intl_calls']

    # Ratio of Total Call Charges to Total Customer Service Calls
    df['charges_service_calls_ratio'] = df['total_call_charges'] / df['number_customer_service_calls']

    # Ratio of Total Call Minutes to Total Customer Service Calls
    df['minutes_service_calls_ratio'] = df['total_call_minutes'] / df['number_customer_service_calls']
    
    # Difference between Total Day Minutes and Total Evening Minutes
    df['day_eve_minutes_difference'] = df['total_day_minutes'] - df['total_eve_minutes']

    # Difference between Total Day Calls and Total Evening Calls
    df['day_eve_calls_difference'] = df['total_day_calls'] - df['total_eve_calls']

    # Difference between Total Day Charges and Total Evening Charges
    df['day_eve_charges_difference'] = df['total_day_charge'] - df['total_eve_charge']

    # Ratio of Total Day Minutes to Total International Minutes
    df['day_intl_minutes_ratio'] = df['total_day_minutes'] / df['total_intl_minutes']

    # Ratio of Total Day Calls to Total International Calls
    df['day_intl_calls_ratio'] = df['total_day_calls'] / df['total_intl_calls']

    # Ratio of Total Day Charges to Total International Charges
    df['day_intl_charges_ratio'] = df['total_day_charge'] / df['total_intl_charge']

    # Difference between Total Evening Minutes and Total Night Minutes
    df['eve_night_minutes_difference'] = df['total_eve_minutes'] - df['total_night_minutes']

    # Difference between Total Evening Calls and Total Night Calls
    df['eve_night_calls_difference'] = df['total_eve_calls'] - df['total_night_calls']

    # Difference between Total Evening Charges and Total Night Charges
    df['eve_night_charges_difference'] = df['total_eve_charge'] - df['total_night_charge']

    # Ratio of Total Evening Minutes to Total International Minutes
    df['eve_intl_minutes_ratio'] = df['total_eve_minutes'] / df['total_intl_minutes']

    # Ratio of Total Evening Calls to Total International Calls
    df['eve_intl_calls_ratio'] = df['total_eve_calls'] / df['total_intl_calls']

    # Ratio of Total Evening Charges to Total International Charges
    df['eve_intl_charges_ratio'] = df['total_eve_charge'] / df['total_intl_charge']

    # Difference between Total Night Minutes and Total International Minutes
    df['night_intl_minutes_difference'] = df['total_night_minutes'] - df['total_intl_minutes']

    # Difference between Total Night Calls and Total International Calls
    df['night_intl_calls_difference'] = df['total_night_calls'] - df['total_intl_calls']

    # Difference between Total Night Charges and Total International Charges
    df['night_intl_charges_difference'] = df['total_night_charge'] - df['total_intl_charge']

    # Ratio of Total Day Minutes to Total Evening Minutes
    df['day_eve_minutes_ratio'] = df['total_day_minutes'] / df['total_eve_minutes']

    # Ratio of Total Day Calls to Total Evening Calls
    df['day_eve_calls_ratio'] = df['total_day_calls'] / df['total_eve_calls']

    # Ratio of Total Day Charges to Total Evening Charges
    df['day_eve_charges_ratio'] = df['total_day_charge'] / df['total_eve_charge']

    # Ratio of Total Day Minutes to Total Night Minutes
    df['day_night_minutes_ratio'] = df['total_day_minutes'] / df['total_night_minutes']

    # Ratio of Total Day Calls to Total Night Calls
    df['day_night_calls_ratio'] = df['total_day_calls'] / df['total_night_calls']

    # Ratio of Total Day Charges to Total Night Charges
    df['day_night_charges_ratio'] = df['total_day_charge'] / df['total_night_charge']

    # Ratio of Total Day Minutes to Total International Minutes
    df['day_intl_minutes_ratio'] = df['total_day_minutes'] / df['total_intl_minutes']

    # Ratio of Total Day Calls to Total International Calls
    df['day_intl_calls_ratio'] = df['total_day_calls'] / df['total_intl_calls']

    # Ratio of Total Day Charges to Total International Charges
    df['day_intl_charges_ratio'] = df['total_day_charge'] / df['total_intl_charge']

    # Ratio of Total Evening Minutes to Total Night Minutes
    df['eve_night_minutes_ratio'] = df['total_eve_minutes'] / df['total_night_minutes']

    # Ratio of Total Evening Calls to Total Night Calls
    df['eve_night_calls_ratio'] = df['total_eve_calls'] / df['total_night_calls']

    # Ratio of Total Evening Charges to Total Night Charges
    df['eve_night_charges_ratio'] = df['total_eve_charge'] / df['total_night_charge']

    # Ratio of Total Evening Minutes to Total International Minutes
    df['eve_intl_minutes_ratio'] = df['total_eve_minutes'] / df['total_intl_minutes']

    # Ratio of Total Evening Calls to Total International Calls
    df['eve_intl_calls_ratio'] = df['total_eve_calls'] / df['total_intl_calls']

    # Ratio of Total Evening Charges to Total International Charges
    df['eve_intl_charges_ratio'] = df['total_eve_charge'] / df['total_intl_charge']

    # Ratio of Total Night Minutes to Total International Minutes
    df['night_intl_minutes_ratio'] = df['total_night_minutes'] / df['total_intl_minutes']

    # Ratio of Total Night Calls to Total International Calls
    df['night_intl_calls_ratio'] = df['total_night_calls'] / df['total_intl_calls']

    # Ratio of Total Night Charges to Total International Charges
    df['night_intl_charges_ratio'] = df['total_night_charge'] / df['total_intl_charge']

    # Total Call Charges per Number of Customer Service Calls
    df['charges_per_service_calls'] = df['total_call_charges'] / df['number_customer_service_calls']

    # Total Call Minutes per Number of Customer Service Calls
    df['minutes_per_service_calls'] = df['total_call_minutes'] / df['number_customer_service_calls']

    # Difference between Total Day Minutes and Total International Minutes
    df['day_intl_minutes_difference'] = df['total_day_minutes'] - df['total_intl_minutes']

    # Difference between Total Day Calls and Total International Calls
    df['day_intl_calls_difference'] = df['total_day_calls'] - df['total_intl_calls']

    # Difference between Total Day Charges and Total International Charges
    df['day_intl_charges_difference'] = df['total_day_charge'] - df['total_intl_charge']

    # Difference between Total Evening Minutes and Total International Minutes
    df['eve_intl_minutes_difference'] = df['total_eve_minutes'] - df['total_intl_minutes']

    # Difference between Total Evening Calls and Total International Calls
    df['eve_intl_calls_difference'] = df['total_eve_calls'] - df['total_intl_calls']

    # Difference between Total Evening Charges and Total International Charges
    df['eve_intl_charges_difference'] = df['total_eve_charge'] - df['total_intl_charge']

    # Difference between Total Night Minutes and Total International Minutes
    df['night_intl_minutes_difference'] = df['total_night_minutes'] - df['total_intl_minutes']

    # Difference between Total Night Calls and Total International Calls
    df['night_intl_calls_difference'] = df['total_night_calls'] - df['total_intl_calls']

    # Difference between Total Night Charges and Total International Charges
    df['night_intl_charges_difference'] = df['total_night_charge'] - df['total_intl_charge']

    # Average Call Duration per Area Code
    df['avg_call_duration_per_area_code'] = df['total_call_minutes'] / df.groupby('area_code')['number_customer_service_calls'].transform('mean')

    # Average Call Duration per State
    df['avg_call_duration_per_state'] = df['total_call_minutes'] / df.groupby('state')['number_customer_service_calls'].transform('mean')

    # Average Call Duration per International Plan
    df['avg_call_duration_per_intl_plan'] = df['total_call_minutes'] / df.groupby('international_plan')['number_customer_service_calls'].transform('mean')

    # Average Call Duration per Voice Mail Plan
    df['avg_call_duration_per_voice_mail_plan'] = df['total_call_minutes'] / df.groupby('voice_mail_plan')['number_customer_service_calls'].transform('mean')

    # Total Charge per Account Length
    df['charge_per_account_length'] = df['total_call_charges'] / df['account_length']

    # Total Minutes per Account Length
    df['minutes_per_account_length'] = df['total_call_minutes'] / df['account_length']

    # Total Charge per Number of Voice Mail Messages
    df['charge_per_vmail_messages'] = df['total_call_charges'] / df['number_vmail_messages']

    # Total Minutes per Number of Voice Mail Messages
    df['minutes_per_vmail_messages'] = df['total_call_minutes'] / df['number_vmail_messages']

    # Average Call Duration per Number of Voice Mail Messages
    df['avg_call_duration_per_vmail_messages'] = df['total_call_minutes'] / df['number_vmail_messages']

    # Total Charge per Number of International Calls
    df['charge_per_intl_calls'] = df['total_call_charges'] / df['total_intl_calls']

    # Total Minutes per Number of International Calls
    df['minutes_per_intl_calls'] = df['total_call_minutes'] / df['total_intl_calls']

    # Average Call Duration per Number of International Calls
    df['avg_call_duration_per_intl_calls'] = df['total_call_minutes'] / df['total_intl_calls']

    # Total Charge per Number of Customer Service Calls
    df['charge_per_service_calls'] = df['total_call_charges'] / df['number_customer_service_calls']

    # Total Minutes per Number of Customer Service Calls
    df['minutes_per_service_calls'] = df['total_call_minutes'] / df['number_customer_service_calls']

    # Average Call Duration per Number of Customer Service Calls
    df['avg_call_duration_per_service_calls'] = df['total_call_minutes'] / df['number_customer_service_calls']

    # Total Charge per Total Call Minutes
    df['charge_per_minutes'] = df['total_call_charges'] / df['total_call_minutes']

    # Total Minutes per Total Call Charges
    df['minutes_per_charge'] = df['total_call_minutes'] / df['total_call_charges']

    # Average Call Duration per Total Call Charges
    df['avg_call_duration_per_charge'] = df['total_call_minutes'] / df['total_call_charges']

    # Average Call Duration per State and Area Code
    df['avg_call_duration_per_state_area'] = df.groupby(['state', 'area_code'])['total_call_minutes'].transform('mean')

    # Average Call Duration per International Plan and Voice Mail Plan
    df['avg_call_duration_per_intl_vmail'] = df.groupby(['international_plan', 'voice_mail_plan'])['total_call_minutes'].transform('mean')

    # Difference between Total Day Minutes and Total Evening Minutes per State
    df['day_eve_minutes_diff_per_state'] = df.groupby('state')['total_day_minutes'].transform(lambda x: x - x.mean())

    # Difference between Total Night Minutes and Total International Minutes per Area Code
    df['night_intl_minutes_diff_per_area'] = df.groupby('area_code')['total_night_minutes'].transform(lambda x: x - x.mean())

    # Ratio of Total Call Charges to Account Length
    df['charge_account_length_ratio'] = df['total_call_charges'] / df['account_length']

    # Ratio of Total Call Minutes to Account Length
    df['minutes_account_length_ratio'] = df['total_call_minutes'] / df['account_length']

    # Ratio of Total Call Charges to Number of Voice Mail Messages
    df['charge_vmail_messages_ratio'] = df['total_call_charges'] / df['number_vmail_messages']

    # Ratio of Total Call Minutes to Number of Voice Mail Messages
    df['minutes_vmail_messages_ratio'] = df['total_call_minutes'] / df['number_vmail_messages']

    # Ratio of Total Call Charges to Number of International Calls
    df['charge_intl_calls_ratio'] = df['total_call_charges'] / df['total_intl_calls']

    # Ratio of Total Call Minutes to Number of International Calls
    df['minutes_intl_calls_ratio'] = df['total_call_minutes'] / df['total_intl_calls']

    # Percentage of Total Call Charges for Day Calls
    df['day_charge_percentage'] = df['total_day_charge'] / df['total_call_charges']

    # Percentage of Total Call Charges for Evening Calls
    df['eve_charge_percentage'] = df['total_eve_charge'] / df['total_call_charges']

    # Percentage of Total Call Charges for Night Calls
    df['night_charge_percentage'] = df['total_night_charge'] / df['total_call_charges']

    # Percentage of Total Call Charges for International Calls
    df['intl_charge_percentage'] = df['total_intl_charge'] / df['total_call_charges']

    # Percentage of Total Call Minutes for Day Calls
    df['day_minutes_percentage'] = df['total_day_minutes'] / df['total_call_minutes']

    # Percentage of Total Call Minutes for Evening Calls
    df['eve_minutes_percentage'] = df['total_eve_minutes'] / df['total_call_minutes']

    # Percentage of Total Call Minutes for Night Calls
    df['night_minutes_percentage'] = df['total_night_minutes'] / df['total_call_minutes']

    # Percentage of Total Call Minutes for International Calls
    df['intl_minutes_percentage'] = df['total_intl_minutes'] / df['total_call_minutes']

    # Percentage of Total Customer Service Calls for Day Calls
    df['day_service_calls_percentage'] = df['total_day_calls'] / df['number_customer_service_calls']

    # Percentage of Total Customer Service Calls for Evening Calls
    df['eve_service_calls_percentage'] = df['total_eve_calls'] / df['number_customer_service_calls']

    # Percentage of Total Customer Service Calls for Night Calls
    df['night_service_calls_percentage'] = df['total_night_calls'] / df['number_customer_service_calls']

    # Percentage of Total Customer Service Calls for International Calls
    df['intl_service_calls_percentage'] = df['total_intl_calls'] / df['number_customer_service_calls']

    # Concatenate the churn_per_state_area column to the original DataFrame
    df = pd.concat([df, df.groupby(['state', 'area_code'])['churn'].mean().rename('churn_per_state_area')], axis=1)

    df = pd.concat([df, df.groupby('state')['number_customer_service_calls'].sum().rename('calls_per_state')], axis=1)
    
def feat_generator(df):
    features = list(set(df.select_dtypes(include='number').columns) - {'churn'})

    # Generate feature transformations
    for feature in features:
        # Square of the feature
        df[f'{feature}_squared'] = df[feature] ** 2

        # Logarithm of the feature (if applicable)
        if df[feature].min() > 0:
            offset = 1e-8  # Adjust the offset value as needed
            df[f'log_{feature}'] = np.log(df[feature] + offset)

        # Square root of the feature (if applicable)
        if df[feature].min() >= 0:
            df[f'sqrt_{feature}'] = np.sqrt(df[feature])

        # Inverse of the feature (if applicable)
        if df[feature].min() > 0:
            df[f'inv_{feature}'] = 1 / df[feature]

    return df


def reduce_multicollinearity(df, feat=None, threshold=0.9):
    # Calculate correlation matrix
    if feat is None:
        corr_matrix = df.select_dtypes(include='number').corr().abs()
    else:
        corr_matrix = df[feat].select_dtypes(include='number').corr().abs()

    # Create a mask to identify highly correlated features
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Find highly correlated feature pairs
    correlated_features = np.where(corr_matrix > threshold, True, False) & mask

    # Identify one feature to remove from each correlated pair
    features_to_remove = set()
    for feature_i, feature_j in zip(*correlated_features.nonzero()):
        if feature_i != feature_j:
            if feat is None:
                column_i = df.columns[feature_i]
                column_j = df.columns[feature_j]
            else:
                column_i = feat[feature_i]
                column_j = feat[feature_j]
            # Add feature with higher correlation to removal set
            if corr_matrix.iloc[feature_i, feature_j] > corr_matrix.iloc[feature_j, feature_i]:
                features_to_remove.add(column_i)
            else:
                features_to_remove.add(column_j)

    # Remove the highly correlated features from the dataframe
    df = df.drop(features_to_remove, axis=1)
    print(features_to_remove)
    return df

def print_corr(df):
    print("Correlation Coefficient of all the Features")
    corr = df.corr()
    corr.sort_values(["churn"], ascending = False, inplace = True)
    correlations = corr.churn
    a = correlations[correlations > 0.1]
    b = correlations[correlations < -0.1]
    top_corr_features = a.append(b)
    print(top_corr_features)

def generate_features(df, feat=None, threshold=0.9, reduce_collinearity=False, heatmap = False):
    """
    Generate the feature matrix by adding additional features, selecting specific features, and reducing multicollinearity.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        feat (list, optional): List of feature names to select. Default is None.
        threshold (float, optional): Threshold value for multicollinearity. Default is 0.9.
        reduce_collinearity (bool, optional): Whether to reduce multicollinearity. Default is False.

    Returns:
        pandas.DataFrame: The generated feature matrix.

    """
    # Add additional features
    added_features(df)
    df = feat_generator(df)
    
    if feat is not None and not reduce_collinearity:
        # Select specific features
        df_reduced = df.loc[:, df.columns.isin(feat + ['churn'])]
    elif reduce_collinearity:
        # Reduce multicollinearity
        df_reduced = reduce_multicollinearity(df, feat, threshold)
    else:
        # Return the original DataFrame
        df_reduced = df
    if heatmap:
        # Perform correlation coefficient ranking
        heatmap_corr(df_reduced)
    
    return df_reduced

def main():
    df, x_train, x_test, y_train, y_test = load_dataset()
    #print(df)
    #plotter(df)
    #scatter(df)
    #Get Correlation of "Churn" with other variables:
    added_features(df)
    feat_generator(df)
    df_reduced = reduce_multicollinearity(df, threshold = 0.1)
    corr_coef_ranking(df_reduced)
    print("Correlation Coefficient of all the Features")
    corr = df.corr()
    corr.sort_values(["churn"], ascending = False, inplace = True)
    correlations = corr.churn
    a = correlations[correlations > 0.1]
    b = correlations[correlations < -0.1]
    top_corr_features = a.append(b)
    print(df_reduced)
    print(top_corr_features)
if __name__ == '__main__':
    main()