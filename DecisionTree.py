import pandas as pd
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

am_rows=1000
output_train_x = pd.read_csv("output_train_x/flights.csv", nrows=am_rows)
output_train_y = pd.read_csv("output_train_y/flights.csv", nrows=am_rows)
output_train_y.loc[output_train_x.index, :]
assert(output_train_y.shape[0] == output_train_x.shape[0])


dictionary = {'max_leaf_nodes': [] , 'max_depth':[] , 'model_score': []}
for leafnodes in range(2,20):
    for md in range(2,20):
        clf = tree.DecisionTreeRegressor(max_leaf_nodes=leafnodes, max_depth=md)
        scores = cross_val_score(clf, output_train_x,output_train_y['ARRIVAL_DELAY'], cv=5)
        dictionary['max_leaf_nodes'].append(leafnodes)
        dictionary['max_depth'].append(md)
        dictionary['model_score'].append(scores.mean())
Decision_Tree_Results=pd.DataFrame(dictionary)
Decision_Tree_Results.to_csv('decisiontree_results/flights.csv')
