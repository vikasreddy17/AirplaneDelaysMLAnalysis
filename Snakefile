rule dataset_split:
	input: 'input_data_df.csv'
	output:'output_train_x/flights.csv', 'output_test_x/flights.csv','output_train_y/flights.csv', 'output_test_y/flights.csv'
	shell: 'python Dataexploration.py --flightsinput_file {input} --flightsoutput_train_x {output[0]} --flightsoutput_test_x {output[1]} --flightsoutput_train_y {output[2]} --flightsoutput_test_y {output[3]}'

rule supportvm:
	input: 'output_train_x/flights.csv', 'output_train_y/flights.csv'
	output: 'supportvm_results/flights.csv'
	shell: 'python SVM_CrossVal.py --input_train_x {input[0]} --input_train_y {input[1]} --supportvm_results {output}'