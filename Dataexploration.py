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
#load data
flightsdata = pd.read_csv("Dataset/flights.csv")
flightsdata = pd.DataFrame(flightsdata)
#variable table
var_discrip_dict = {'Variable Name': [] , 'Description':[] , 'Catagorical/Continuous': []}
var_discrip_dict['Variable Name'] = ['YEAR', 'MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE', 'FLIGHT_NUMBER', 'TAIL_NUMBER', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'DEPARTURE_DELAY', 'TAXI_OUT', 'WHEELS_OFF', 'SCHEDULED_TIME', 'ELAPSED_TIME', "AIR_TIME", 'DISTANCE', 'WHEELS_ON', 'TAXI_IN', "SCHEDULED_ARRIVAL", 'ARRIVAL_TIME', 'ARRIVAL_DELAY', 'DIVERTED', 'CANCELLED', 'CANCELLATION_REASON', 'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY']
var_discrip_dict['Description'] = ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1',]
var_discrip_dict['Catagorical/Continuous'] = ['Catagorical', 'Catagorical', 'Catagorical', 'Catagorical', 'Catagorical', 'Catagorical', 'ID', 'Catagorical', 'Catagorical', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 'Catagorical', 'Catagorical', 'Catagorical', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 'Continuous']
var_discrip_dict = pd.DataFrame(var_discrip_dict)

continuous_varsdf = (var_discrip_dict.loc[var_discrip_dict['Catagorical/Continuous'] == 'Continuous', :])
catagorical_varsdf = (var_discrip_dict.loc[var_discrip_dict['Catagorical/Continuous'] == 'Catagorical', :])


#Univariate analysis
flightsdata.hist()
plt.savefig('FULL_flights_histogram.png')
#bivariate analysis
corrmatrix = flightsdata.corr()
sea.heatmap(corrmatrix, annot=True)
plt.savefig('FULL_correlation_heatmap.png')

flightsdata = flightsdata.loc[flightsdata['CANCELLED'] == 0,:]
#missing values
flightsdata = flightsdata[['SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'DEPARTURE_DELAY', 'TAXI_OUT', 'WHEELS_OFF', 'SCHEDULED_TIME', 'ELAPSED_TIME', 'AIR_TIME', 'DISTANCE', 'WHEELS_ON', 'TAXI_IN', 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME', 'ARRIVAL_DELAY']]
flightsdata = flightsdata.loc[flightsdata.isna().sum(axis=1) == 0, :]
print(flightsdata.shape)

target_flightsdata = flightsdata[['ARRIVAL_DELAY']]
cont_flightsdata = flightsdata[['SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'DEPARTURE_DELAY', 'TAXI_OUT', 'WHEELS_OFF', 'SCHEDULED_TIME', 'ELAPSED_TIME', 'AIR_TIME', 'DISTANCE', 'WHEELS_ON', 'TAXI_IN', 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME']]
x_train, x_test, y_train, y_test = train_test_split(cont_flightsdata, target_flightsdata, test_size=0.2, random_state=100)
x_train.to_csv('output_train_x/flights.csv' ,index=None)
x_test.to_csv('output_test_x/flights.csv' ,index=None)
y_train.to_csv('output_train_y/flights.csv' ,index=None)
y_test.to_csv('output_test_y/flights.csv',index=None)
