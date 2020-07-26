import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import svm

am_rows = 1000
output_train_x = pd.read_csv("output_train_x/flights.csv", nrows=am_rows)
output_train_y = pd.read_csv("output_train_y/flights.csv", nrows=am_rows)

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
SVM_results=pd.DataFrame(dictionary)
SVM_results.to_csv(supportvm_results/flights.csv)