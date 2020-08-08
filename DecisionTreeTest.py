import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
import pdb
from sklearn.metrics import r2_score
import os

#load in data
decisiontree_cross_val_results = pd.read_csv("DecisionTree_full_crossval_results.csv")
decisiontree_cross_val_results.sort_values(by=['test_score'], inplace=True, ascending=False)
decisiontree_cross_val_results.to_csv('DecisionTree_full_crossval_results.csv', index=None)
output_test_x = pd.read_csv("output_data/output_test_x.csv")
output_test_y = pd.read_csv("output_data/output_test_y.csv")
output_train_x = pd.read_csv("output_data/output_train_x.csv")
output_train_y = pd.read_csv("output_data/output_train_y.csv")

#test and score
clf = tree.DecisionTreeRegressor(max_leaf_nodes=26, max_depth=19)
clf.fit(output_train_x,output_train_y['ARRIVAL_DELAY'])
predict_val = clf.predict(output_test_x)
r2_score = r2_score(predict_val, output_test_y)
print('decision tree test score using r-squared metric is')
print(r2_score)

#create full dataframe for testing scores from the various models
model_test_scores = {'model': [], 'model_test_scores': []}
model_test_scores['model'].append('Decision Tree')
model_test_scores['model_test_scores'].append(r2_score)
model_test_scores = pd.DataFrame(model_test_scores)
if os.path.isfile('FinalModelScores.csv') == False:
	model_test_scores.to_csv('FinalModelScores.csv', index=None)
else:
	exist_model_test_scores = pd.read_csv('model_test_scores.csv')
	exist_model_test_scores = pd.concat([exist_model_test_scores, model_test_scores], axis=0)
	exist_model_test_scores.to_csv('FinalModelScores.csv')
