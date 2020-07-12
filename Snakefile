rule dataset_split:
	input: 'input_data_df.csv'
	output:'output_train_x/flights.csv', 'output_test_x/flights.csv','output_train_y/flights.csv', 'output_test_y/flights.csv'
	shell: 'python Dataexploration.py --flightsinput_file {input} --flightsoutput_train_x {output[0]} --flightsoutput_test_x {output[1]} --flightsoutput_train_y {output[2]} --flightsoutput_test_y {output[3]}'