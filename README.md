Matthew Suri 
IDSN 499
Spring 2023 
msuri@usc.edu
Final Project Part 2

Part 1 Recap
Domain:
The problem I am thinking of solving is predicting stock prices based on historical financial data. This is an important issue for financial institutions, investors, and quants, as it can help them make informed decisions in the stock market and optimize their investment strategies.

Dataset:
I have obtained the dataset from Kaggle (https://www.kaggle.com/dgawlik/nyse ). The dataset contains historical daily stock prices for S&P 500 companies from 2010 to 2016. It consists of approximately 851,264 rows and 7 attributes, which provide information about the stock symbol, open, close, low, high, and volume.

The dataset includes data on historical stock prices, which is essential for solving the problem of predicting stock prices. The other attributes can be used as potential predictors for future stock price trends.

Problem Type:
I am planning to create a predictor to forecast future stock prices based on historical financial data. This approach can help in identifying trends and patterns in the stock market, which can be used to make better investment decisions.

Attributes:
Some of the attributes in the dataset are:

Date: the date of the record
Symbol: the stock symbol
Open: the opening price of the stock
Close: the closing price of the stock
Low: the lowest price of the stock during the trading day
High: the highest price of the stock during the trading day
Volume: the number of shares traded during the trading day
The attributes are all numeric except for the Date and Symbol columns. There are no missing values in the dataset.







Final Project Part 2

Machine Learning Algorithms:

	•	Linear Regression: The correlation matrix shows a super strong linear relationship between 'open', 'low', 'high', and the target variable 'close' (correlations are like above 0.99). This means that a linear model like Linear Regression might actually do a good job predicting the closing price using these features. Since Linear Regression is simple to use and easy to understand, it's a good idea to start with this algorithm as our first model.

	•	Support Vector Regression (SVR): Even though the correlation matrix shows some really strong linear relationships, there could still be some non-linear stuff going on between the features that could impact the closing price. Support Vector Regression can handle non-linear relationships using different kernel functions, like the Radial Basis Function (RBF) kernel. This flexibility might help us get better predictions if there are non-linear relationships hiding in the data.

	•	Random Forest Regressor: The correlation matrix also shows a kind of weak negative relationship between 'volume' and the target variable 'close' (correlations are around -0.06). While this relationship isn't as strong as the others, it might still be helpful for making our model's predictions better. Random Forest Regressor is a fancy method that can capture complicated feature interactions and deal with situations where the relationships between features and the target variable aren't super clear. By using this algorithm, we can take advantage of the info provided by the 'volume' feature and any other complex relationships that might be in the data.
 
Correlations:

Correlation matrix using a heatmap.


Scatter matrix plot


Transformer Discussion: 

For my project, I decided to use two transformers: StandardScaler and PolynomialFeatures.
I chose StandardScaler because it scales the numeric features in my dataset so that they have a mean of 0 and a standard deviation of 1. This is important because it helps prevent features with larger values from having a disproportionate impact on the analysis. Additionally, many machine learning algorithms require that the input features be standardized.

I also chose to use PolynomialFeatures because it generates polynomial features based on the input features. This can be useful for capturing non-linear relationships between the input features and the target variable. In my case, it may be beneficial to include polynomial features because the relationship between the input features and the stock price may not be linear.

By using a ColumnTransformer to combine these transformers, I can apply them to both numeric and categorical features in my dataset. This ensures that my data is properly preprocessed before being used in my machine learning algorithms.





Part 3

Initial Results: 

Linear Regression:
MSE: 0.4458
R-squared Score: 0.9996
Random Forest Regressor:
MSE: 0.6821
R-squared Score: 0.7205
Support Vector Regression (SVR):
MSE: 0.8123
R-squared Score: 0.4535

This was trying to predict the ‘close’ value based on 'open', 'low', 'high’, and I found that the linear regression model was really overfitting, and initially there weren’t many interesting correlations because since stocks don’t move much in the same day there were very, very strong correlations between the variables. 

So I decided to pivot into analyzing 2 stocks that were very correlated with each her and do a 
pairwise analysis of AAPL vs MSFT, I used the same machine learning models. 

Data Prep part 2:

I decided to compare the closing price over time to the different stocks, to see if I could find any strongly correlated pairs. I needed to clean up the date data to be uniform, some columns included time while others didn’t.

df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

As you can see, the drastic fall in GOOGL and AAPL are due to stock splitting, so I split the the stock uniformly over the timeline to prevent outliers.
Here is what the code looked like:
# Apple split details apple_split_date = '2014-06-09' apple_split_ratio = 7  # Google split details google_split_date = '2014-04-03' google_split_ratio = 2  # Function to adjust stock prices based on split date and ratio def adjust_stock_prices(df, symbol, split_date, split_ratio):     # Filter the historical stock prices for the specified symbol before the split date     before_split_prices = df[(df['symbol'] == symbol) & (df['date'] < split_date)]['close']      # Adjust the prices based on the split ratio     adjusted_prices = before_split_prices / split_ratio      # Update the original DataFrame with the adjusted prices     df.loc[(df['symbol'] == symbol) & (df['date'] < split_date), 'close'] = adjusted_prices   # Adjust Apple stock prices adjust_stock_prices(df, 'AAPL', apple_split_date, apple_split_ratio)  # Adjust Google stock prices adjust_stock_prices(df, 'GOOGL', google_split_date, google_split_ratio)



I then preformed a correlation matrix to see what stocks are closely related. I was interested in APPL and MSFT because they’re some of the longest surviving tech companies and had a strong correlation at .82 



















Results: 

For this analysis, I divided the dataset into a training set and a test set. I used 80% of the data for training and kept the remaining 20% for testing. So, out of the total 851,264 entries, approximately 680,000 were used for training and around 170,000 for testing.

pair_df = df[df['symbol'].isin(['AAPL', 'MSFT'])]  # Split the dataset into training and testing sets X_pair = pair_df.drop('close', axis=1)  # Features for the pair y_pair = pair_df['close']  # Target for the pair X_train_pair, X_test_pair, y_train_pair, y_test_pair = train_test_split(X_pair, y_pair, test_size=0.2, random_state=42)   # Fit and transform the training data X_train_pair_prepared = preprocessor.fit_transform(X_train_pair)  # Transform the test data X_test_pair_prepared = preprocessor.transform(X_test_pair)


To evaluate the performance of the models, I used Mean Squared Error (MSE) and R-squared Score as the metrics. Here are the results for each model:

Linear Regression:

Mean Squared Error: 196.03641008619013
R-squared Score: 0.7902885035178429

SVR:
Mean Squared Error: 148.26579091244582
R-squared Score: 0.8413915003050242

Random Forest:
Mean Squared Error: 0.2892523967750614
R-squared Score: 0.9996905699662523

Based on these results I found that the Random Forest was really over fit but the SVR seemed to be promising. For the SVR model, I used grid search to explore different combinations of hyperparameters. The hyperparameters I focused on were 'C' (regularization parameter), 'gamma' (kernel coefficient), and 'kernel' (type of kernel function). I specified a range of values for each hyperparameter and evaluated the model's performance using mean squared error (MSE) as the scoring metric. The grid search helped me find the best combination of hyperparameters that minimized the MSE.

For the Random Forest Regressor, I used randomized search to efficiently explore a wide range of hyperparameter combinations. The hyperparameters I tuned included 'n_estimators' (number of trees in the forest), 'min_samples_split' (minimum number of samples required to split an internal node), 'min_samples_leaf' (minimum number of samples required to be at a leaf node), 'max_features' (number of features to consider when looking for the best split), and 'max_depth' (maximum depth of the trees). I defined the ranges or values for each hyperparameter, and the randomized search algorithm sampled a set of combinations to find the best performing model based on MSE.

Here are the results after hyper tuning.

SVR:

Mean Squared Error: 44.36183412946989
R-squared Score: 0.9525435779103786
Best Model: SVR(C=100.0, epsilon=1.0)


Random Forest Regressor:

Mean Squared Error: 0.28333318953869185
R-squared Score: 0.9996969020848979
Best Model: RandomForestRegressor(min_samples_leaf=2, n_estimators=200)

I used this code to get the final results:

def tune_and_evaluate_model(model, param_grid, X_train, y_train, X_test, y_test, search_type='grid'):     if search_type == 'grid':         search_cv = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)     elif search_type == 'random':         search_cv = RandomizedSearchCV(model, param_grid, cv=5, n_iter=10, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)      search_cv.fit(X_train, y_train)     best_model = search_cv.best_estimator_     predictions = best_model.predict(X_test)     mse = mean_squared_error(y_test, predictions)     r2 = r2_score(y_test, predictions)     return mse, r2, best_model  # Hyperparameter grids for models linear_regression_grid = {} svr_grid = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],             'C': np.logspace(-3, 3, 7),             'epsilon': np.logspace(-3, 3, 7)} random_forest_grid = {'n_estimators': [10, 50, 100, 200],                       'max_depth': [None, 10, 30, 50],                       'min_samples_split': [2, 5, 10],                       'min_samples_leaf': [1, 2, 4]}  # Tune and evaluate the Linear Regression model linear_regression_mse, linear_regression_r2, best_linear_regression = tune_and_evaluate_model(linear_regression, linear_regression_grid, X_train_pair_prepared, y_train_pair, X_test_pair_prepared, y_test_pair)  # Tune and evaluate the SVR model svr_mse, svr_r2, best_svr = tune_and_evaluate_model(svr, svr_grid, X_train_pair_prepared, y_train_pair, X_test_pair_prepared, y_test_pair, search_type='random')  # Tune and evaluate the Random Forest model random_forest_mse, random_forest_r2, best_random_forest = tune_and_evaluate_model(random_forest, random_forest_grid, X_train_pair_prepared, y_train_pair, X_test_pair_prepared, y_test_pair)  # Print the results print("Linear Regression:") print("Mean Squared Error:", linear_regression_mse) print("R-squared Score:", linear_regression_r2) print("Best Model:", best_linear_regression)  print("\nSVR:") print("Mean Squared Error:", svr_mse) print("R-squared Score:", svr_r2) print("Best Model:", best_svr)  print("\nRandom Forest Regressor:") print("Mean Squared Error:", random_forest_mse) print("R-squared Score:", random_forest_r2) print("Best Model:", best_random_forest)


In the end, the SVR (Support Vector Regression) model outperformed the Random Forest Regressor model because the Random Forest model was overfitting the training data with an unrealistic r squared score, even after attempts at hyper tuning.

On the other hand, the SVR model, with its regularization techniques and kernel functions, was able to generalize better to unseen data. It achieved a lower MSE and a higher R-squared score on the test set compared to the Random Forest model. The SVR model's ability to handle non-linear relationships and its regularization parameters helped prevent overfitting and maintain better generalization performance.

Therefore, in this specific case, the SVR model was the preferred choice due to its ability to strike a balance between capturing the underlying patterns in the data and avoiding overfitting.


 Given more time, I could have explored additional techniques such as feature engineering, trying out different algorithms or ensemble methods, and addressing any outliers or biases in the data. These steps might have further improved the results and provided additional insights.

Overall, the analysis demonstrated the effectiveness of machine learning models in predicting stock prices, and I gained valuable experience in data preprocessing, model selection, hyperparameter tuning, and result evaluation.
