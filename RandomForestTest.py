import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import os


#load in data
randomforest_cross_val_results = pd.read_csv("RandomForest_full_crossval_results.csv")
randomforest_cross_val_results.sort_values(by=['test_score'], inplace=True, ascending=False)
randomforest_cross_val_results.to_csv('RandomForest_full_crossval_results.csv', index=None)
output_test_x = pd.read_csv("output_data/output_test_x.csv")
output_test_y = pd.read_csv("output_data/output_test_y.csv")
output_train_x = pd.read_csv("output_data/output_train_x.csv")
output_train_y = pd.read_csv("output_data/output_train_y.csv")

#test and score
clf = RandomForestRegressor(n_estimators=41, max_depth=4, random_state=0)
clf.fit(output_train_x,output_train_y['ARRIVAL_DELAY'])
predict_val = clf.predict(output_test_x)
r2_score = r2_score(predict_val, output_test_y)
print('random forest test score using r-squared metric is')
print(r2_score)

#create full dataframe for testing scores from the various models
randomforest_cross_val_best_results = randomforest_cross_val_results.loc[randomforest_cross_val_results['n_estimators'] == 41,:]
randomforest_cross_val_best_results = randomforest_cross_val_best_results.loc[randomforest_cross_val_best_results['max_depth'] == 4,:]
model_test_scores = {'model': [], 'model_test_scores': []}
model_test_scores['model'].append('Random Forest')
model_test_scores['model_test_scores'].append(r2_score)
model_test_scores = pd.DataFrame(model_test_scores)
model_test_scores['fit_time'] = randomforest_cross_val_best_results['fit_time']
model_test_scores['crossval_train_score'] = randomforest_cross_val_best_results['train_score']
model_test_scores['crossval_test_score'] = randomforest_cross_val_best_results['test_score']
model_test_scores['crossval_train_minus_test'] = randomforest_cross_val_best_results['train_minus_test']
model_test_scores['crossval_train_minus_test'] = randomforest_cross_val_best_results['train_minus_test']
if os.path.isfile('FinalModelScores.csv') == False:
	model_test_scores.to_csv('FinalModelScores.csv', index=None)
else:
	exist_model_test_scores = pd.read_csv('FinalModelScores.csv')
	exist_model_test_scores = pd.concat([exist_model_test_scores, model_test_scores], axis=0)
	exist_model_test_scores.to_csv('FinalModelScores.csv', index=None)