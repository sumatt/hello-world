import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from transformer3 import CustomPolynomialFeatures

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# Load dataset
df = pd.read_csv('prices.csv')

# Print attributes and the number of values each attribute has
print(df.info())

# Convert 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

# Apple split details
apple_split_date = '2014-06-09'
apple_split_ratio = 7

# Google split details
google_split_date = '2014-04-03'
google_split_ratio = 2

# Function to adjust stock prices based on split date and ratio
def adjust_stock_prices(df, symbol, split_date, split_ratio):
    # Filter the historical stock prices for the specified symbol before the split date
    before_split_prices = df[(df['symbol'] == symbol) & (df['date'] < split_date)]['close']

    # Adjust the prices based on the split ratio
    adjusted_prices = before_split_prices / split_ratio

    # Update the original DataFrame with the adjusted prices
    df.loc[(df['symbol'] == symbol) & (df['date'] < split_date), 'close'] = adjusted_prices


# Adjust Apple stock prices
adjust_stock_prices(df, 'AAPL', apple_split_date, apple_split_ratio)

# Adjust Google stock prices
adjust_stock_prices(df, 'GOOGL', google_split_date, google_split_ratio)

# Select the top 10 tech companies
top_tech_companies = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB', 'TSLA', 'NVDA', 'INTC', 'ADBE', 'CSCO']
subset_df = df[df['symbol'].isin(top_tech_companies)]

# Pivot the filtered dataset
pivot_df = subset_df.pivot_table(index='date', columns='symbol', values='close')

# Calculate the correlation matrix
correlation_matrix = pivot_df.corr()
print(correlation_matrix)

# # Create the heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
# plt.title('Correlation Matrix of Top Tech Companies')
# plt.show()

# # Plot the stock prices over time
# pivot_df.plot(figsize=(10, 6))
# plt.xlabel('Date')
# plt.ylabel('Closing Price')
# plt.title('Stock Prices Over Time')
# plt.show()

# Identify the categorical columns
categorical_columns = ['date', 'symbol']
numeric_columns = ['open', 'low', 'high', 'volume']

# Prepare the dataset for training
X = df.drop('close', axis=1)  # Removing the target column
y = df['close']

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create transformers for numeric and categorical features
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('poly', CustomPolynomialFeatures())
])
categorical_transformer = OneHotEncoder()

# Identify the categorical column (only the 'symbol' column in this case)
categorical_columns = ['symbol']

# # Perform correlation analysis on numeric columns
# correlation_matrix2 = df[numeric_columns].corr()
# print(correlation_matrix2)

# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])

# # Fit and transform the training data
# X_train_prepared = preprocessor.fit_transform(X_train)
#
# # Transform the test data
# X_test_prepared = preprocessor.transform(X_test)

# # Visualize the correlation matrix using a heatmap
# plt.figure(figsize=(8, 6))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
# plt.show()

# # Create scatter matrix plot
# scatter_matrix(df[numeric_columns], figsize=(12, 8))
# plt.show()

# Prepare the dataset for pairwise analysis (AAPL and MSFT)
pair_df = df[df['symbol'].isin(['AAPL', 'MSFT'])]

# Split the dataset into training and testing sets
X_pair = pair_df.drop('close', axis=1)  # Features for the pair
y_pair = pair_df['close']  # Target for the pair
X_train_pair, X_test_pair, y_train_pair, y_test_pair = train_test_split(X_pair, y_pair, test_size=0.2, random_state=42)


# Fit and transform the training data
X_train_pair_prepared = preprocessor.fit_transform(X_train_pair)

# Transform the test data
X_test_pair_prepared = preprocessor.transform(X_test_pair)

# Create the three ML models
linear_regression = LinearRegression()
svr = SVR()
random_forest = RandomForestRegressor()

# # Linear Regression)
# pair_model = LinearRegression()
#
# # Train the model
# pair_model.fit(X_train_pair_prepared, y_train_pair)
#
# # Make predictions on the testing set
# pair_predictions = pair_model.predict(X_test_pair_prepared)
#
# # Evaluate the model
# pair_mse = mean_squared_error(y_test_pair, pair_predictions)
# pair_r2 = r2_score(y_test_pair, pair_predictions)
#
# # Print the results
# print("Pairwise Analysis (AAPL vs MSFT):")
# print("Mean Squared Error:", pair_mse)
# print("R-squared Score:", pair_r2)

# # SVR
# pair_model = SVR()
#
# # Train the model
# pair_model.fit(X_train_pair_prepared, y_train_pair)
#
# # Make predictions on the testing set
# pair_predictions = pair_model.predict(X_test_pair_prepared)
#
# # Evaluate the model
# pair_mse = mean_squared_error(y_test_pair, pair_predictions)
# pair_r2 = r2_score(y_test_pair, pair_predictions)
#
# # Print the results
# print("Pairwise Analysis (AAPL vs MSFT) - SVR:")
# print("Mean Squared Error:", pair_mse)
# print("R-squared Score:", pair_r2)
#
# # Random Forest
# pair_model = RandomForestRegressor()
#
# # Train the model
# pair_model.fit(X_train_pair_prepared, y_train_pair)
#
# # Make predictions on the testing set
# pair_predictions = pair_model.predict(X_test_pair_prepared)
#
# # Evaluate the model
# pair_mse = mean_squared_error(y_test_pair, pair_predictions)
# pair_r2 = r2_score(y_test_pair, pair_predictions)
#
# # Print the results
# print("Pairwise Analysis (AAPL vs MSFT) - Random Forest:")
# print("Mean Squared Error:", pair_mse)
# print("R-squared Score:", pair_r2)


# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.ensemble import RandomForestRegressor
#
# # Define the parameter distributions to sample from
# param_dist = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [None, 5, 10],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
# }
#
# # Create the random forest model
# rf_model = RandomForestRegressor()
#
# # Create the random search object
# random_search = RandomizedSearchCV(
#     estimator=rf_model,
#     param_distributions=param_dist,
#     n_iter=10,  # Number of parameter settings to sample
#     scoring='neg_mean_squared_error',
#     cv=5,  # Number of cross-validation folds
#     random_state=42
# )
#
# # Fit the random search to the training data
# random_search.fit(X_train_pair_prepared, y_train_pair)
#
# # Get the best hyperparameters and model
# best_params = random_search.best_params_
# best_model = random_search.best_estimator_
#
# # Make predictions on the testing set
# pair_predictions = best_model.predict(X_test_pair_prepared)
#
# # Evaluate the model
# pair_mse = mean_squared_error(y_test_pair, pair_predictions)
# pair_r2 = r2_score(y_test_pair, pair_predictions)
#
# # Print the results
# print("Pairwise Analysis (AAPL vs MSFT) - Random Forest:")
# print("Best Hyperparameters:", best_params)
# print("Mean Squared Error:", pair_mse)
# print("R-squared Score:", pair_r2)

# SVR hyper tuning

# # Define the parameter grid
# param_grid = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
#              'C': np.logspace(-3, 3, 7),
#              'epsilon': np.logspace(-3, 3, 7)}
#
# # Create the SVR model
# svr_model = SVR()
#
# # Perform Grid Search with cross-validation
# grid_search = GridSearchCV(estimator=svr_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
# grid_search.fit(X_train_pair_prepared, y_train_pair)
#
# # Get the best hyperparameters
# best_params = grid_search.best_params_
#
# # Train the SVR model with the best hyperparameters
# best_svr_model = SVR(**best_params)
# best_svr_model.fit(X_train_pair_prepared, y_train_pair)
#
# # Make predictions
# svr_predictions = best_svr_model.predict(X_test_pair_prepared)
#
# # Evaluate the model
# svr_mse = mean_squared_error(y_test_pair, svr_predictions)
# svr_r2 = r2_score(y_test_pair, svr_predictions)

# Print the results
# print("SVR - Best Hyperparameters:", best_params)
# print("Mean Squared Error:", svr_mse)
# print("R-squared Score:", svr_r2)


def tune_and_evaluate_model(model, param_grid, X_train, y_train, X_test, y_test, search_type='grid'):
    if search_type == 'grid':
        search_cv = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    elif search_type == 'random':
        search_cv = RandomizedSearchCV(model, param_grid, cv=5, n_iter=10, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)

    search_cv.fit(X_train, y_train)
    best_model = search_cv.best_estimator_
    predictions = best_model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mse, r2, best_model

# Hyperparameter grids for models
linear_regression_grid = {}
svr_grid = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'C': np.logspace(-3, 3, 7),
            'epsilon': np.logspace(-3, 3, 7)}
random_forest_grid = {'n_estimators': [10, 50, 100, 200],
                      'max_depth': [None, 10, 30, 50],
                      'min_samples_split': [2, 5, 10],
                      'min_samples_leaf': [1, 2, 4]}

# Tune and evaluate the Linear Regression model
linear_regression_mse, linear_regression_r2, best_linear_regression = tune_and_evaluate_model(linear_regression, linear_regression_grid, X_train_pair_prepared, y_train_pair, X_test_pair_prepared, y_test_pair)

# Tune and evaluate the SVR model
svr_mse, svr_r2, best_svr = tune_and_evaluate_model(svr, svr_grid, X_train_pair_prepared, y_train_pair, X_test_pair_prepared, y_test_pair, search_type='random')

# Tune and evaluate the Random Forest model
random_forest_mse, random_forest_r2, best_random_forest = tune_and_evaluate_model(random_forest, random_forest_grid, X_train_pair_prepared, y_train_pair, X_test_pair_prepared, y_test_pair)

# Print the results
print("Linear Regression:")
print("Mean Squared Error:", linear_regression_mse)
print("R-squared Score:", linear_regression_r2)
print("Best Model:", best_linear_regression)

print("\nSVR:")
print("Mean Squared Error:", svr_mse)
print("R-squared Score:", svr_r2)
print("Best Model:", best_svr)

print("\nRandom Forest Regressor:")
print("Mean Squared Error:", random_forest_mse)
print("R-squared Score:", random_forest_r2)
print("Best Model:", best_random_forest)

# Fit the SVR model on the training data
# best_svr.fit(X_train_pair_prepared, y_train_pair)
#
# # Make predictions on the test data
# svr_predictions = best_svr.predict(X_test_pair_prepared)

# # Plot the actual values and predicted values
# plt.figure(figsize=(10, 6))
# plt.scatter(X_test_pair['date'], y_test_pair, color='blue', label='Actual')
# plt.scatter(X_test_pair['date'], svr_predictions, color='red', label='Predicted')
# plt.xlabel('Date')
# plt.ylabel('Closing Price')
# plt.title('SVR Predictions vs Actual')
# plt.legend()
# plt.show()


# # Create a scatter plot of actual values vs predicted values
# plt.figure(figsize=(10, 6))
# plt.scatter(y_test_pair, svr_predictions, color='blue', alpha=0.5)
# plt.plot([min(y_test_pair), max(y_test_pair)], [min(y_test_pair), max(y_test_pair)], color='red', linestyle='--')
# plt.xlabel('Actual Values')
# plt.ylabel('SVR Predicted Values')
# plt.title('Actual Values vs SVR Predicted Values')
# plt.show()
