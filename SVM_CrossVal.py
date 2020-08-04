import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import svm
from tqdm import tqdm

print('start')
am_rows=1000
output_train_x = pd.read_csv("output_data/output_train_x.csv")
output_train_y = pd.read_csv("output_data/output_train_y.csv")
output_train_y.loc[output_train_x.index, :]
assert(output_train_y.shape[0] == output_train_x.shape[0])

dictionary = {'kernelname': [] , 'cvalue':[] , 'Score': []}
for ker in tqdm(['rbf', 'poly', 'linear']):
    for val in tqdm(range(5,20)):
        clf = svm.SVR(kernel=ker, C=(val/1000))
        scores = cross_val_score(clf, output_train_x, output_train_y['ARRIVAL_DELAY'], cv=5, scoring='r2')
        dictionary['kernelname'].append(ker)
        dictionary['cvalue'].append(val/1000)
        dictionary['recallval'].append(scores.mean())
SVM_results=pd.DataFrame(dictionary)
SVM_results.to_csv('SVM_cross_val_results.csv')
print("done")