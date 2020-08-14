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