# AirplaneDelaysMLAnalysis
Look through code flow chart and final research paper to get an overview of project.

Machine learning techniques can be utilized to predict delays in aircraft departure and arrival times. The study uses regression models to predict flight delay as a continuous variable. The models (decision tree regressor, random forest regressor, and AdaBoost regressor) were chosen from the sklearn python library.

Flight data was taken from the US Bureau of Transportation Statistics. A subset of one million samples was randomly selected from the full data set. These samples were then split into a training and testing set. Using the training data, a five-fold cross-validation was run on the three models selected to optimize the hyperparameters. Then, the three models used the input features from the testing set and predicted the arrival delay of each flight. The models’ predictions for the testing set were compared to the actual arrival delays. The accuracy of each of the models was scored using the R-squared metric.

After training and testing the machine learning models, my research focused on comparing different levels of complex machine learning algorithms. Theoretically, the more complex algorithms (random forests and AdaBoost) should have performed better than the base algorithm (decision tree), but this was not the case. All three models had very similar testing scores. However, with model fit time in consideration, the simple decision tree model is of much greater value.
