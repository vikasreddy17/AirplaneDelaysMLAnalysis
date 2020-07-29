rule Dataexploration:
	input: 'input_data_df.csv'
	output:'output_train_x/flights.csv', 'output_test_x/flights.csv','output_train_y/flights.csv', 'output_test_y/flights.csv'
	shell: 'python Dataexploration.py'

rule SVM_CrossVal:
	input: 'output_train_x/flights.csv', 'output_train_y/flights.csv'
	output: 'supportvm_results.csv'
	shell: 'python SVM_CrossVal.py'

rule DecisionTree:
	input: 'output_train_x/flights.csv', 'output_train_y/flights.csv'
	output: 'decisiontree_results.csv'
	shell: 'python DecisionTree.py'

rule AdaBoost:
	input: 'output_train_x/flights.csv', 'output_train_y/flights.csv'
	output: 'AdaBoost_results.csv'
	shell: 'python DecisionTree.py'

rule Random:
	input: 'AdaBoost_results.csv', 'decisiontree_results.csv' ,'supportvm_results.csv'
	output: 'random.csv'
	shell: 'python SVM_CrossVal.py', 'python DecisionTree.py', 'python AdaBoost.py'
