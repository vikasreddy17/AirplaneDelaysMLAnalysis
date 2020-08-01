rule Dataexploration:
	input: 'input_data_df.csv'
	output:'output_data/output_train_x.csv', 'output_data/output_test_x.csv','output_data/output_train_y.csv', 'output_data/output_test_y.csv'
	shell: 'python Dataexploration.py'

rule SVM_CrossVal:
	input: 'output_data/output_train_x.csv', 'output_data/output_train_y.csv'
	output: 'supportvm_results.csv'
	shell: 'python SVM_CrossVal.py'

rule DecisionTree:
	input: 'output_data/output_train_x.csv', 'output_data/output_train_y.csv'
	output: 'decisiontree_results.csv'
	shell: 'python DecisionTree.py'

rule RandomForest:
	input: 'output_data/output_train_x.csv', 'output_data/output_train_y.csv'
	output: 'RandomForest_results.csv'
	shell: 'python RandomForest.py'

rule AdaBoost:
	input: 'output_data/output_train_x.csv', 'output_data/output_train_y.csv'
	output: 'AdaBoost_results.csv'
	shell: 'python AdaBoost.py'

rule Random:
	input: 'AdaBoost_results.csv', 'decisiontree_results.csv', 'RandomForest_results.csv'
	output: 'random.csv'
	shell: 'echo done > {output}'
