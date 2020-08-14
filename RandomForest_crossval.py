import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_validate
from tqdm import tqdm
import os

print('start')
output_train_x = pd.read_csv("output_data/output_train_x.csv")
output_train_y = pd.read_csv("output_data/output_train_y.csv")
output_train_y.loc[output_train_x.index, :]
assert(output_train_y.shape[0] == output_train_x.shape[0])

RandomForest_Full_Results = None
for n_estimators in tqdm(range(42, 47)):
    for md in tqdm(range(3,5)):
        if os.path.isfile('RandomForest_crossval/' + str(n_estimators) + str(md) + '.csv') == False:
            clf1 = RandomForestRegressor(n_estimators=n_estimators, max_depth=md, random_state=0)
            scores = cross_validate(clf1, output_train_x,output_train_y['ARRIVAL_DELAY'], cv=5, scoring='r2', return_train_score=True)
            dictionary = {'n_estimators': [] , 'max_depth':[], 'test_score': [], 'train_score': [], 'fit_time': [], 'train_minus_test': []}
            dictionary['n_estimators'].append(n_estimators)
            dictionary['max_depth'].append(md)
            dictionary['test_score'].append(scores['test_score'].mean())
            dictionary['train_score'].append(scores['train_score'].mean())
            dictionary['fit_time'].append(scores['fit_time'].mean())
            dictionary['train_minus_test'].append((scores['train_score'].mean()) - (scores['test_score'].mean()))
            RandomForest_Results=pd.DataFrame(dictionary)
            RandomForest_Results.to_csv('RandomForest_crossval/' + str(n_estimators) + str(md) + '.csv', index=None)
        else:
            RandomForest_Results = pd.read_csv('RandomForest_crossval/' + str(n_estimators) + str(md) + '.csv')
        if RandomForest_Full_Results is None:
            RandomForest_Full_Results = RandomForest_Results
        else:
            RandomForest_Full_Results = pd.concat([RandomForest_Full_Results, RandomForest_Results], axis=0)
RandomForest_Full_Results.to_csv('RandomForest_full_crossval_results.csv', index=None)
print('done')