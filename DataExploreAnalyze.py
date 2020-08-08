#import packages
import pandas as pd
import seaborn as sea
import sklearn as skl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn import svm
import pdb
from tqdm import tqdm
import os

#load data and subset
flightsdata = pd.read_csv("Dataset/flights.csv")
flightsdata = pd.DataFrame(flightsdata)
flightsdata = flightsdata.sample(2000000)

#variable table
var_discrip_dict = {'Variable Name': [] , 'Description':[] , 'Catagorical/Continuous': []}
var_discrip_dict['Variable Name'] = ['YEAR', 'MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE', 'FLIGHT_NUMBER', 'TAIL_NUMBER', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'DEPARTURE_DELAY', 'TAXI_OUT', 'WHEELS_OFF', 'SCHEDULED_TIME', 'ELAPSED_TIME', "AIR_TIME", 'DISTANCE', 'WHEELS_ON', 'TAXI_IN', "SCHEDULED_ARRIVAL", 'ARRIVAL_TIME', 'ARRIVAL_DELAY', 'DIVERTED', 'CANCELLED', 'CANCELLATION_REASON', 'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY']
var_discrip_dict['Description'] = ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1',]
var_discrip_dict['Catagorical/Continuous'] = ['Catagorical', 'Catagorical', 'Catagorical', 'Catagorical', 'Catagorical', 'Catagorical', 'ID', 'Catagorical', 'Catagorical', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 'Catagorical', 'Catagorical', 'Catagorical', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 'Continuous']
var_discrip_dict = pd.DataFrame(var_discrip_dict)

#bivariate analysis
corrmatrix = flightsdata.corr()
sea.heatmap(corrmatrix, annot=True)
plt.savefig('Charts/FULL_correlation_heatmap.png')

#Univariate analysis
flightsdata.hist()
plt.savefig('Charts/FULL_flights_histogram.png')

#correlation bar chart
def imp_clean_corr_data():
    if os.path.isfile('Charts/correlation_matrix.csv') == False:
      corrmatrix = flightsdata.corr()
      sea.heatmap(correlation_matrix, annot=False)
      corrmatrix = corrmatrix.stack()
      corrmatrix = corrmatrix.reset_index()
      corrmatrix = corrmatrix.rename(columns={'level_0': 'feature_1', 'level_1': 'feature_2', '0': 'correlation'})
      corrmatrix.to_csv('correlation_matrix.csv')
    else:
      corrmatrix = pd.read_csv('Charts/correlation_matrix.csv')
      corrmatrix = corrmatrix.rename(columns={'level_0': 'feature_1', 'level_1': 'feature_2'})
      corrmatrix.sort_values(by=['correlation'], inplace=True, ascending=False)
    list_of_indexes = []
    for i in range(0,587):
      list_of_indexes.append(i)
    corrmatrix = corrmatrix.replace(to_replace=corrmatrix.index, value=list_of_indexes, inplace=False, limit=None, regex=False, method='pad')
    corrmatrix = corrmatrix.loc[corrmatrix['correlation'] != 1, :]
    corrmatrix = corrmatrix.loc[corrmatrix['correlation'] != 253.0, :]
    cleaned_correlation_matrix = None
    list_of_want_indexes = []
    for i in range(0,587,2):
      list_of_want_indexes.append(i)
    for index in list_of_want_indexes:
        temp_correlation_matrix = corrmatrix.loc[corrmatrix['index'] == index, :]
        cleaned_correlation_matrix = pd.concat([cleaned_correlation_matrix, temp_correlation_matrix], axis=0)
    cleaned_correlation_matrix.sort_values(by=['correlation'], inplace=True, ascending=False)
    return cleaned_correlation_matrix
def corr_bar_chart(file):
	corrmatrix = pd.read_csv(file)
	corrmatrix = corrmatrix.head(20)
	corrmatrix['feature_both'] = corrmatrix['feature_1'] + " and " + corrmatrix['feature_2']
	corrmatrix.drop(columns=['feature_1', 'feature_2', 'Unnamed: 0', 'index'])
	corrmatrix.plot(kind='barh', x='feature_both', y='correlation')
	plt.savefig('Charts/flights_correlation_bar_chart.png')
corrmatrix = imp_clean_corr_data()
corrmatrix.to_csv('Charts/cleaned_airplane_correlation_table.csv')
corr_bar_chart('Charts/cleaned_airplane_correlation_table.csv')



#missing values treatment and data cleaning
flightsdata = flightsdata.loc[flightsdata['CANCELLED'] == 0,:]
flightsdata = flightsdata[['SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'DEPARTURE_DELAY', 'TAXI_OUT', 'WHEELS_OFF', 'SCHEDULED_TIME', 'ELAPSED_TIME', 'AIR_TIME', 'DISTANCE', 'WHEELS_ON', 'TAXI_IN', 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME', 'ARRIVAL_DELAY']]
flightsdata = flightsdata.loc[flightsdata.isna().sum(axis=1) == 0, :]


target_flightsdata = flightsdata[['ARRIVAL_DELAY']]
cont_flightsdata = flightsdata[['SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'DEPARTURE_DELAY', 'TAXI_OUT', 'WHEELS_OFF', 'SCHEDULED_TIME', 'ELAPSED_TIME', 'AIR_TIME', 'DISTANCE', 'WHEELS_ON', 'TAXI_IN', 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME']]
if os.path.isfile('output_data/output_train_x.csv') == False or os.path.isfile('output_data/output_train_y.csv') == False or os.path.isfile('output_data/output_test_x.csv') == False or os.path.isfile('output_data/output_test_y.csv') == False:
	x_train, x_test, y_train, y_test = train_test_split(cont_flightsdata, target_flightsdata, test_size=0.2, random_state=100)
	x_train.to_csv('output_data/output_train_x.csv' ,index=None)
	x_test.to_csv('output_data/output_test_x.csv' ,index=None)
	y_train.to_csv('output_data/output_train_y.csv' ,index=None)
	y_test.to_csv('output_data/output_test_y.csv',index=None)










