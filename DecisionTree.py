import pandas as pd
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeRegressor
import pdb
from tqdm import tqdm

print('start')
am_rows=1000
output_train_x = pd.read_csv("output_data/output_train_x.csv")
output_train_y = pd.read_csv("output_data/output_train_y.csv")
output_train_y.loc[output_train_x.index, :]
assert(output_train_y.shape[0] == output_train_x.shape[0])


dictionary = {'max_leaf_nodes': [] , 'max_depth':[] , 'test_score': [], 'train_score': []}
for leafnodes in tqdm(range(2,20)):
    for md in tqdm(range(2,20)):
        clf = tree.DecisionTreeRegressor(max_leaf_nodes=leafnodes, max_depth=md)
        scores = cross_validate(clf, output_train_x,output_train_y['ARRIVAL_DELAY'], cv=5, scoring='r2', return_train_score=True)
        dictionary['max_leaf_nodes'].append(leafnodes)
        dictionary['max_depth'].append(md)
        dictionary['test_score'].append(scores['test_score'].mean())
        dictionary['train_score'].append(scores['train_score'].mean())
Decision_Tree_Results=pd.DataFrame(dictionary)
Decision_Tree_Results.to_csv('decisiontree_cross_val_results.csv')
print('done')
