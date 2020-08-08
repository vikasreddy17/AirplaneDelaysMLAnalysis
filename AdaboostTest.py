import pandas as pd
from sklearn import tree
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import r2_score
import os

adaboost_cross_val_results = pd.read_csv("AdaBoost_full_crossval_results.csv")
adaboost_cross_val_results['train_minus_train'] = adaboost_cross_val_results['train_score'] - adaboost_cross_val_results['test_score']
adaboost_cross_val_results.sort_values(by=['test_score'], inplace=True, ascending=False)
adaboost_cross_val_results.to_csv('AdaBoost_full_crossval_results.csv', index=None)
output_test_x = pd.read_csv("output_data/output_test_x.csv")
output_test_y = pd.read_csv("output_data/output_test_y.csv")
output_train_x = pd.read_csv("output_data/output_train_x.csv")
output_train_y = pd.read_csv("output_data/output_train_y.csv")

#test and score
clf = tree.DecisionTreeRegressor(max_depth=4)
clf1 = AdaBoostRegressor(base_estimator=clf, n_estimators=25, learning_rate=(89/1000), random_state=0)
clf.fit(output_train_x,output_train_y['ARRIVAL_DELAY'])
predict_val = clf.predict(output_test_x)
r2_score = r2_score(predict_val, output_test_y)
print('AdaBoost test score using r-squared metric is')
print(r2_score)

#create full dataframe for testing scores from the various models
model_test_scores = {'model': [], 'model_test_scores': []}
model_test_scores['model'].append('AdaBoost')
model_test_scores['model_test_scores'].append(r2_score)
model_test_scores = pd.DataFrame(model_test_scores)
if os.path.isfile('model_test_scores.csv') == False:
	model_test_scores.to_csv('FinalModelScores.csv', index=None)
else:
	exist_model_test_scores = pd.read_csv('model_test_scores.csv')
	exist_model_test_scores = pd.concat([exist_model_test_scores, model_test_scores], axis=0)
	exist_model_test_scores.to_csv('FinalModelScores.csv')

