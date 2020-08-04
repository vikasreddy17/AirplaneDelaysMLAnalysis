import pandas as pd
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_validate
from tqdm import tqdm
import os

print('start')
am_rows=1000
output_train_x = pd.read_csv("output_data/output_train_x.csv", nrows=am_rows)
output_train_y = pd.read_csv("output_data/output_train_y.csv", nrows=am_rows)
output_train_y.loc[output_train_x.index, :]
assert(output_train_y.shape[0] == output_train_x.shape[0])

RandomForest_Full_Results = None
for n_estimators in tqdm(range(19, 100)):
    for crit in tqdm(['mse', 'mae']):
    	for md in tqdm(range(2,6)):
            if os.path.isfile('random_forest_crossval/' + str(n_estimators) + str(md) + str(crit) + '.csv') == False:
                clf1 = RandomForestRegressor(n_estimators=n_estimators, criterion=crit, max_depth=md, random_state=0)
                scores = cross_validate(clf1, output_train_x,output_train_y['ARRIVAL_DELAY'], cv=5, scoring='r2', return_train_score=True)
                dictionary = {'n_estimators': [] , 'criterion':[], 'max_depth':[], 'test_score': [], 'train_score': [], 'fit_time': []}
                dictionary['n_estimators'].append(n_estimators)
                dictionary['max_depth'].append(md)
                dictionary['criterion'].append(crit)
                dictionary['test_score'].append(scores['test_score'].mean())
                dictionary['train_score'].append(scores['train_score'].mean())
                dictionary['fit_time'].append(scores['fit_time'])
                RandomForest_Results=pd.DataFrame(dictionary)
                RandomForest_Results.to_csv('random_forest_crossval/' + str(n_estimators) + str(md) + str(crit) + '.csv', index=None)
            else:
                RandomForest_Results = pd.read_csv('random_forest_crossval/' + str(n_estimators) + str(md) + str(crit) + '.csv')
            if RandomForest_Full_Results is None:
                RandomForest_Full_Results = RandomForest_Results
            else:
                RandomForest_Full_Results = pd.concat([RandomForest_Full_Results, RandomForest_Results], axis=0)
RandomForest_Full_Results.to_csv('RandomForest_Full_Results.csv', index=None)

print('done')
