rule Dataexploration:
	input: 'input_data_df.csv'
	output:'output_data/output_train_x.csv', 'output_data/output_test_x.csv','output_data/output_train_y.csv', 'output_data/output_test_y.csv'
	shell: 'python Dataexploration.py'

rule DecisionTree_crossval:
	input: 'output_data/output_train_x.csv', 'output_data/output_train_y.csv'
	output: 'DecisionTree_full_crossval_results.csv'
	shell: 'python DecisionTree_crossval.py'

rule RandomForest_crossval:
	input: 'output_data/output_train_x.csv', 'output_data/output_train_y.csv'
	output: 'RandomForest_full_crossval_results.csv'
	shell: 'python RandomForest_crossval.py'

rule AdaBoost_crossval:
	input: 'output_data/output_train_x.csv', 'output_data/output_train_y.csv'
	output: 'AdaBoost_full_crossval_results.csv'
	shell: 'python AdaBoost_crossval.py'

rule all_crossval
	input: 'output_data/output_train_x.csv', 'output_data/output_test_x.csv','output_data/output_train_y.csv', 'output_data/output_test_y.csv','DecisionTree_full_crossval_results.csv', 'RandomForest_full_crossval_results.csv', 'AdaBoost_full_crossval_results.csv'
	output: 'all_crossval_snake.csv'
	shell: 'echo done > {output}'