#import packages
import pandas as pd
import seaborn as sea
import sklearn as skl
import matplotlib.pyplot as plt
import argparse
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn import svm
parser = argparse.ArgumentParser()
parser.add_argument('--flightsinput_file', type=str, help='snakefile', default=None)
parser.add_argument('--flightsoutput_train_x', type=str, help='snakefile', default=None)
parser.add_argument('--flightsoutput_test_x', type=str, help='snakefile', default=None)
parser.add_argument('--flightsoutput_train_y', type=str, help='snakefile', default=None)
parser.add_argument('--flightsoutput_test_y', type=str, help='snakefile', default=None)
args = parser.parse_args()
print(args)
#load data
flightsdata = pd.read_csv("Dataset/flights.csv")
flightsdata = pd.DataFrame(flightsdata)
#variable table
var_discrip_dict = {'Variable Name': [] , 'Description':[] , 'Catagorical/Continuous': []}
for varname in ['YEAR', 'MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE', 'FLIGHT_NUMBER', 'TAIL_NUMBER', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'DEPARTURE_DELAY', 'TAXI_OUT', 'WHEELS_OFF', 'SCHEDULED_TIME', 'ELAPSED_TIME', "AIR_TIME", 'DISTANCE', 'WHEELS_ON', 'TAXI_IN', "SCHEDULED_ARRIVAL", 'ARRIVAL_TIME', 'ARRIVAL_DELAY', 'DIVERTED', 'CANCELLED', 'CANCELLATION_REASON', 'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY']:
	var_discrip_dict['Variable Name'].append(varname)
for descrip in ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1',]:
	var_discrip_dict['Description'].append(descrip)
for vtype in [['Catagorical', 'Catagorical', 'Catagorical', 'Catagorical', 'Catagorical', 'Catagorical', 'ID', 'Catagorical', 'Catagorical', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 'Catagorical', 'Catagorical', 'Catagorical', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 'Continuous']]:
	var_discrip_dict['Catagorical/Continuous'].append(vtype)
var_discrip_dict = pd.DataFrame(var_discrip_dict)
#Univariate analysis
flightsdata.hist()
#bivariate analysis
corrmatrix = flightsdata.corr()
sea.heatmap(corrmatrix, annot=True)

continuous_varsdf = (var_discrip_dict.loc[var_discrip_dict['Catagorical/Continuous'] == 'Continuous', :])
catagorical_varsdf = (var_discrip_dict.loc[var_discrip_dict['Catagorical/Continuous'] == 'Catagorical', :])

cont_flightsdata = flightsdata[['SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'DEPARTURE_DELAY', 'TAXI_OUT', 'WHEELS_OFF', 'SCHEDULED_TIME', 'ELAPSED_TIME', 'AIR_TIME', 'DISTANCE', 'WHEELS_ON', 'TAXI_IN', 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME', 'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY']]
target_flightsdata = flightsdata['ARRIVAL_DELAY']
x_train, x_test, y_train, y_test = train_test_split(cont_flightsdata, target_flightsdata, test_size=0.2,random_state=0)
x_train.to_csv(args.flightsoutput_train_x)
x_test.to_csv(args.flightsoutput_test_x)
y_train.to_csv(args.flightsoutput_train_y)
y_test.to_csv(args.flightsoutput_test_y)
