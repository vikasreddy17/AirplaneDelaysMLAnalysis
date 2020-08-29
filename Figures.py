import pandas as pd
import matplotlib.pyplot as plt
import os


decisiontree_cross_val_results = pd.read_csv("DecisionTree_full_crossval_results.csv")
decisiontree_cross_val_results = decisiontree_cross_val_results[['test_score', 'fit_time']]
decisiontree_cross_val_results['model'] = 'Decision Tree'
randomforest_cross_val_results = pd.read_csv("RandomForest_full_crossval_results.csv")
randomforest_cross_val_results = randomforest_cross_val_results[['test_score', 'fit_time']]
randomforest_cross_val_results['model'] = 'Random Forest'
adaboost_cross_val_results = pd.read_csv("AdaBoost_full_crossval_results.csv")
adaboost_cross_val_results = adaboost_cross_val_results[['test_score', 'fit_time']]
adaboost_cross_val_results['model'] = 'AdaBoost'

if os.path.isfile('all_model_crossval_results.csv') == False:
  all_model_results = None
  all_model_results = decisiontree_cross_val_results
  all_model_results = pd.concat([all_model_results, randomforest_cross_val_results], axis=0)
  all_model_results = pd.concat([all_model_results, adaboost_cross_val_results], axis=0)
  all_model_results.to_csv('all_model_crossval_results.csv')
else:
  all_model_results = pd.read_csv('all_model_crossval_results.csv')


fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(decisiontree_cross_val_results['test_score'], decisiontree_cross_val_results['fit_time'], s=100, c='tab:green', marker="o", label='Decision Tree')
ax1.scatter(randomforest_cross_val_results['test_score'], randomforest_cross_val_results['fit_time'], s=100, c='tab:blue', marker="o", label='Random Forest')
ax1.scatter(adaboost_cross_val_results['test_score'], adaboost_cross_val_results['fit_time'], s=100, c='tab:orange', marker="o", label='AdaBoost')
plt.legend(loc='upper left')
plt.xlabel('Cross Validation Test Scores (r-squared metric)', fontsize=17)
plt.ylabel('Model Fit Time (seconds)', fontsize=17)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.savefig('Charts/fit_time_scatterplot.png')

randomforest_cross_val_results = pd.read_csv("RandomForest_full_crossval_results.csv")
adaboost_cross_val_results = pd.read_csv("AdaBoost_full_crossval_results.csv")

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(randomforest_cross_val_results['n_estimators'], randomforest_cross_val_results['fit_time'], s=100, c='tab:blue', marker="o", label='Random Forest')
plt.legend(loc='upper left')
plt.xlabel('Number of Weak Learners')
plt.ylabel('Model Fit Time (seconds)')
plt.savefig('Charts/randomforest_weak_learers_fit_time_scatterplot.png')

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(adaboost_cross_val_results['n_estimators'], adaboost_cross_val_results['fit_time'], s=100, c='tab:orange', marker="o", label='AdaBoost')
plt.legend(loc='upper left')
plt.xlabel('Number of Weak Learners')
plt.ylabel('Model Fit Time (seconds)')
plt.savefig('Charts/adaboost_weak_learers_fit_time_scatterplot.png')




