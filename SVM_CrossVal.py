import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn import svm
import argparse


output_train_x = pd.read_csv("output_train_x/flights.csv", nrows=100)
output_train_y = pd.read_csv("output_train_y/flights.csv", nrows=100)

parser = argparse.ArgumentParser()
parser.add_argument('--input_train_x', type=str, help='snakefile', default=None)
parser.add_argument('--input_train_y', type=str, help='snakefile', default=None)
parser.add_argument('--supportvm_results', type=str, help='snakefile', default=None)
args = parser.parse_args()


output_train_y.loc[output_train_x.index, :]
assert(output_train_y.shape[0] == output_train_x.shape[0])
dictionary = {'kernelname': [] , 'cvalue':[] , 'recallval': []}
for ker in ['rbf', 'poly', 'linear']:
    for val in range(5,20):
        clf = svm.SVC(kernel=ker, C=(val/1000))
        scores = cross_val_score(clf, output_train_x, output_train_y, cv=5, scoring='recall_macro')
        dictionary['kernelname'].append(ker)
        dictionary['cvalue'].append(val/1000)
        dictionary['recallval'].append(scores.mean())
new_dataframe=pd.DataFrame(dictionary)
new_dataframe.to_csv(args.supportvm_results)