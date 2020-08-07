import pandas as pd

decisiontree_cross_val_results = pd.read_csv("decisiontree_cross_val_results.csv")
train_minus_test = decisiontree_cross_val_results["train_score"] - decisiontree_cross_val_results["test_score"]
train_minus_test.loc[decisiontree_cross_val_results.index, :]
assert(train_minus_test.shape[0] == decisiontree_cross_val_results.shape[0])
DataFrame.to_dict(decisiontree_cross_val_results)
DataFrame.to_dict(train_minus_test)
