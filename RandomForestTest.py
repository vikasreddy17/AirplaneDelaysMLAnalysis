import pandas as pd

randomforest_cross_val_results = pd.read_csv("RandomForest_Full_Results.csv")
randomforest_cross_val_results.sort_values(by=['test_score'], inplace=True, ascending=False)
randomforest_cross_val_results.to_csv('RandomForest_Full_Results.csv')