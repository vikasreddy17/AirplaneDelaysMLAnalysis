import pandas as pd
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification

am_rows=1000
output_train_x = pd.read_csv("output_train_x/flights.csv", nrows=am_rows)
output_train_y = pd.read_csv("output_train_y/flights.csv", nrows=am_rows)
output_train_y.loc[output_train_x.index, :]
assert(output_train_y.shape[0] == output_train_x.shape[0])


dictionary = {'n_estimators': [] , 'learning_rate':[], 'max_depth':[], 'model_score': []}
for n_estimators in range(10,30):
    for md in range(2,5):
    	for learning_rate in range(85, 95):
        	clf = tree.DecisionTreeRegressor(max_depth=md)
        	dictionary['n_estimators'].append(n_estimators)
        	dictionary['max_depth'].append(md)
        	dictionary['learning_rate'].append(learning_rate/100)
        	clf1 = AdaBoostClassifier(base_estimator=clf, n_estimators=n_estimators, learning_rate=(learning_rate/100), random_state=0)
        	scores = cross_val_score(clf1, output_train_x,output_train_y['ARRIVAL_DELAY'], cv=5)
        	dictionary['model_score'].append(scores.mean())
AdaBoost_Results=pd.DataFrame(dictionary)
AdaBoost_Results.to_csv('AdaBoost_results.csv')