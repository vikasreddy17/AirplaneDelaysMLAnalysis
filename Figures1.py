import pandas as pd
import matplotlib.pyplot as plt
import pickle as pik
from sklearn.metrics import r2_score
import numpy

output_test_x = pd.read_csv("output_data/output_test_x.csv")
output_test_y = pd.read_csv("output_data/output_test_y.csv")

clf = pik.load(open('best_AdaBoost_model.pickle','rb'))
predict_val = clf.predict(output_test_x)
predict_val = pd.DataFrame(predict_val)
test_predictions = output_test_x
test_predictions['Arrival Delay Model Prediction'] = predict_val
test_pred_act = pd.concat([test_predictions,output_test_y['ARRIVAL_DELAY']], axis=1)
test_pred_act.sort_values(by=['MONTH'], inplace=True, ascending=False)
test_pred_act = test_pred_act.round(3)
AdaBoost_dictionary_month_scores = {}
for month in range(1,13):
	test_pred_act = test_pred_act.loc[test_pred_act['MONTH'] == month,:]
	predictions = test_pred_act['Arrival Delay Model Prediction']
	actual = test_pred_act['ARRIVAL_DELAY']
	r2_score = r2_score(actual, predictions)

	if month == 1:
		AdaBoost_jan_r2_score = r2_score
		AdaBoost_dictionary_month_scores['Month'] = 'January'
		AdaBoost_dictionary_month_scores['R-squared Score'] = AdaBoost_jan_r2_score
	elif month == 2:
		AdaBoost_feb_r2_score = r2_score
		AdaBoost_dictionary_month_scores['Month'] = 'January'
		AdaBoost_dictionary_month_scores['R-squared Score'] = AdaBoost_feb_r2_score
	elif month == 3:
		AdaBoost_mar_r2_score = r2_score
		AdaBoost_dictionary_month_scores['Month'] = 'January'
		AdaBoost_dictionary_month_scores['R-squared Score'] = AdaBoost_mar_r2_score
	elif month == 4:
		AdaBoost_apr_r2_score = r2_score
		AdaBoost_dictionary_month_scores['Month'] = 'January'
		AdaBoost_dictionary_month_scores['R-squared Score'] = AdaBoost_apr_r2_score
	elif month == 5:
		AdaBoost_may_r2_score = r2_score
		AdaBoost_dictionary_month_scores['Month'] = 'January'
		AdaBoost_dictionary_month_scores['R-squared Score'] = AdaBoost_may_r2_score
	elif month == 6:
		AdaBoost_jun_r2_score = r2_score
		AdaBoost_dictionary_month_scores['Month'] = 'January'
		AdaBoost_dictionary_month_scores['R-squared Score'] = AdaBoost_jun_r2_score
	elif month == 7:
		AdaBoost_jul_r2_score = r2_score
		AdaBoost_dictionary_month_scores['Month'] = 'January'
		AdaBoost_dictionary_month_scores['R-squared Score'] = AdaBoost_jul_r2_score
	elif month == 8:
		AdaBoost_aug_r2_score = r2_score
		AdaBoost_dictionary_month_scores['Month'] = 'January'
		AdaBoost_dictionary_month_scores['R-squared Score'] = AdaBoost_aug_r2_score
	elif month == 9:
		AdaBoost_sep_r2_score = r2_score
		AdaBoost_dictionary_month_scores['Month'] = 'January'
		AdaBoost_dictionary_month_scores['R-squared Score'] = AdaBoost_sep_r2_score
	elif month == 10:
		AdaBoost_oct_r2_score = r2_score
		AdaBoost_dictionary_month_scores['Month'] = 'January'
		AdaBoost_dictionary_month_scores['R-squared Score'] = AdaBoost_oct_r2_score
	elif month == 11:
		AdaBoost_nov_r2_score = r2_score
		AdaBoost_dictionary_month_scores['Month'] = 'January'
		AdaBoost_dictionary_month_scores['R-squared Score'] = AdaBoost_nov_r2_score
	elif month == 12:
		AdaBoost_dec_r2_score = r2_score
		AdaBoost_dictionary_month_scores['Month'] = 'January'
		AdaBoost_dictionary_month_scores['R-squared Score'] = AdaBoost_dictionary_month_scores
AdaBoost_dictionary_month_scores['Model'] = 'AdaBoost'
AdaBoost_dictionary_month_scores.to_csv('lol.csv')
