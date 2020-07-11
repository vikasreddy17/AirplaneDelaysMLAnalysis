#import packages
import pandas as pd
import seaborn as sea
import sklearn as skl
import matplotlib.pyplot as plt
#load data
flightsdata = pd.read_csv("Dataset/flights.csv")
flightsdata = pd.DataFrame(flightsdata)
#variable table
var_discrip_dict = {'Variable Name': [] , 'Description':[] , 'Catagorical/Continuous': []}
for varname in ['YEAR', 'MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE', 'FLIGHT_NUMBER', 'TAIL_NUMBER', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'DEPARTURE_DELAY', 'TAXI_OUT', 'WHEELS_OFF', 'SCHEDULED_TIME', 'ELAPSED_TIME', "AIR_TIME", 'DISTANCE', 'WHEELS_ON', 'TAXI_IN', "SCHEDULED_ARRIVAL", 'ARRIVAL_TIME', 'ARRIVAL_DELAY', 'DIVERTED', 'CANCELLED', 'CANCELLATION_REASON', 'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY']:
	var_discrip_dict['Variable Name'].append(varname)
for descrip in ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1',]:
	var_discrip_dict['Description'].append(descrip)
for vtype in ['Catagorical', 'Catagorical', 'Catagorical', 'Catagorical', 'Catagorical', 'Catagorical', 'ID', 'Catagorical', 'Catagorical', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 'Catagorical', 'Catagorical', 'Catagorical', 'Continuous', 'Continuous', 'Continuous', 'Continuous', 'Continuous']:
	var_discrip_dict['Catagorical/Continuous'].append(vtype)
var_discrip_dict = pd.DataFrame(var_discrip_dict)
#print(var_discrip_dict)
#Univariate analysis
#flightsdata.hist()
#bivariate analysis
corrmatrix = flightsdata.corr()
sea.heatmap(corrmatrix, annot=True)
plt.show()