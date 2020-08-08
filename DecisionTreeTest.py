import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor

decisiontree_cross_val_results = pd.read_csv("Decision_Tree_Full_Results.csv")
decisiontree_cross_val_results.sort_values(by=['test_score'], inplace=True, ascending=False)
decisiontree_cross_val_results.to_csv('Decision_Tree_Full_Results.csv')
output_test_x = pd.read_csv("output_data/output_test_x.csv")
output_test_y = pd.read_csv("output_data/output_test_y.csv")


clf = tree.DecisionTreeRegressor(max_leaf_nodes=26, max_depth=19)
clf.fit(output_test_x,output_test_y)