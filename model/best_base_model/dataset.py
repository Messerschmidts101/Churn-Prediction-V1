from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split


def transform(df):
    '''
    Returns df, x_train, x_test, y_train, y_test
    '''
    one_hot_encoder(df)
    target = df['churn']
    features = df.drop('churn', axis=1)

    df['feature_cross'] = df['international_plan'] & df['voice_mail_plan']
    print(df['feature_cross'])

    # Splitting the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    print(features.columns)
    desc(df)
    return df, x_train, x_test, y_train, y_test

def one_hot_encoder(df):
    '''
    Converts the categorical values to numerical 
    '''
    label_encoder = LabelEncoder()

    # Encode categorical columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = label_encoder.fit_transform(df[col])

def desc(df):
    # number of churn and non-churn 
    counts = df.churn.value_counts()
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

def load_dataset():
    import pandas as pd
    
    df = pd.read_csv('data/train.csv')
    return transform(df)