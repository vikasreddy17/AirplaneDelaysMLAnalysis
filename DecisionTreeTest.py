import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
import pdb
from sklearn.metrics import r2_score
import os
from sklearn.model_selection import cross_validate
import pickle as pik
import numpy as np
import matplotlib.pyplot as plt
import graphviz

#load in data
decisiontree_cross_val_results = pd.read_csv("DecisionTree_full_crossval_results.csv")
output_test_x = pd.read_csv("output_data/output_test_x.csv")
output_test_y = pd.read_csv("output_data/output_test_y.csv")
output_train_x = pd.read_csv("output_data/output_train_x.csv")
output_train_y = pd.read_csv("output_data/output_train_y.csv")

#test and score
clf = tree.DecisionTreeRegressor(max_leaf_nodes=26, max_depth=19)
clf.fit(output_train_x,output_train_y['ARRIVAL_DELAY'])
dot_data = tree.export_graphviz(clf, out_file='DecisionTree.dot')
graph = graphviz.Source(dot_data) 

pik.dump(clf, open( 'best_DecisionTree_model.pickle','wb'))
predict_val = clf.predict(output_test_x)
r2_score = r2_score(predict_val, output_test_y)
print('decision tree test score using r-squared metric is')
print(r2_score)

#create full dataframe for testing scores from the various models
decisiontree_cross_val_best_test = decisiontree_cross_val_results.loc[decisiontree_cross_val_results['max_leaf_nodes'] == 26,:]
decisiontree_cross_val_best_test = decisiontree_cross_val_best_test.loc[decisiontree_cross_val_best_test['max_depth'] == 19,:]
model_test_scores = {'model': [], 'model_test_scores': [], 'fit_time': []}
model_test_scores['model'].append('Decision Tree')
model_test_scores['model_test_scores'].append(r2_score)
clf = tree.DecisionTreeRegressor(max_leaf_nodes=26, max_depth=19)
scores = cross_validate(clf, output_train_x,output_train_y['ARRIVAL_DELAY'], cv=5, scoring='r2', return_train_score=True)
model_test_scores['fit_time'].append(scores['fit_time'].mean())
model_test_scores = pd.DataFrame(model_test_scores)
model_test_scores['crossval_train_score'] = decisiontree_cross_val_best_test['train_score']
model_test_scores['crossval_test_score'] = decisiontree_cross_val_best_test['test_score']
model_test_scores['crossval_train_minus_test'] = decisiontree_cross_val_best_test['train_minus_test']

if os.path.isfile('FinalModelScores.csv') == False:
	model_test_scores.to_csv('FinalModelScores.csv', index=None)
else:
	exist_model_test_scores = pd.read_csv('FinalModelScores.csv')
	exist_model_test_scores = pd.concat([exist_model_test_scores, model_test_scores], axis=0)
	exist_model_test_scores.to_csv('FinalModelScores.csv', index=None)

