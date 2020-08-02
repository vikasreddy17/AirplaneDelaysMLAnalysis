import pandas as pd
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_validate
from tqdm import tqdm

print('start')
am_rows=1000
output_train_x = pd.read_csv("output_data/output_train_x.csv")
output_train_y = pd.read_csv("output_data/output_train_y.csv")
output_train_y.loc[output_train_x.index, :]
assert(output_train_y.shape[0] == output_train_x.shape[0])


dictionary = {'n_estimators': [] , 'learning_rate':[], 'test_score': [], 'train_score': [], 'max_depth': []}
for n_estimators in tqdm(range(10,30)):
    for learning_rate in range(85, 95):
    	for md in range(2,6):
    		clf = tree.DecisionTreeRegressor(max_depth=md)
    		clf1 = AdaBoostRegressor(base_estimator=clf, n_estimators=n_estimators, learning_rate=(learning_rate/100), random_state=0)
    		scores = cross_validate(clf1, output_train_x,output_train_y['ARRIVAL_DELAY'], cv=5, scoring='r2', return_train_score=True)
    		dictionary['n_estimators'].append(n_estimators)
    		dictionary['max_depth'].append(md)
    		dictionary['learning_rate'].append(learning_rate/100)
    		dictionary['test_score'].append(scores['test_score'].mean())
    		dictionary['train_score'].append(scores['train_score'].mean())
AdaBoost_Results=pd.DataFrame(dictionary)
AdaBoost_Results.to_csv('AdaBoost_cross_val_results.csv')
print('done')