import pandas as pd

adaboost_cross_val_results = pd.read_csv("AdaBoost_Full_Results.csv")



adaboost_cross_val_results.sort_values(by=['test_score'], inplace=True, ascending=False)
adaboost_cross_val_results.to_csv('AdaBoost_Full_Results.csv')