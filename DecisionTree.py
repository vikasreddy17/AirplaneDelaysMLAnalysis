import pandas as pd
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import argparse

am_rows=30
output_train_x = pd.read_csv("output_train_x/flights.csv", nrows=am_rows)
output_train_y = pd.read_csv("output_train_y/flights.csv", nrows=am_rows)

parser = argparse.ArgumentParser()
parser.add_argument('--input_train_x', type=str, help='snakefile', default=None)
parser.add_argument('--input_train_y', type=str, help='snakefile', default=None)
parser.add_argument('--decisiontree_results', type=str, help='snakefile', default=None)
args = parser.parse_args()


X = [[0, am_rows], [0, 13]]
Y = [0, am_rows]

output_train_y.loc[output_train_x.index, :]
assert(output_train_y.shape[0] == output_train_x.shape[0])
dictionary = {'max_leaf_nodes': [] , 'max_depth':[] , 'model_score': []}
for leafnodes in range(20):
    for md in range(20):
        clf = tree.DecisionTreeClassifier(max_leaf_nodes=leafnodes, max_depth=md)
        cross_val_score(clf, output_train_x,output_train_y, cv=5)
        dictionary['max_leaf_nodes'].append(leafnodes)
        dictionary['max_depth'].append(md)
        dictionary['model_score'].append(scores.mean(self, X, Y, sample_weight=None))
Decision_Tree_Results=pd.DataFrame(dictionary)
Decision_Tree_Results.to_csv(args.decisiontree_results)