'''
new feature candidates:
1. avg_duration_per_day_calls
2. 
'''
import pandas as pd

orig_data_set = pd.read_csv('model/data/dataset.csv')
feats_data_set = pd.read_csv('model/data/feature_ranking_support.csv')
feats_columns = feats_data_set[feats_data_set['Ranking'] <= 2].sort_values('Ranking', ascending=True)
#feats_columns = feats_data_set['Feature']
print(feats_columns)
"""
print("AAAAA check if orig columns exists in feats columns AAAAA")
for column in orig_data_set.columns:
    if column in feats_columns['Feature'].to_list():
        print(column)"""

print("\n AAAAA check if orig columns exists in feats columns AAAAA")
for column in feats_columns['Feature'].to_list():
    if column not in orig_data_set.columns:
        print(column)
    