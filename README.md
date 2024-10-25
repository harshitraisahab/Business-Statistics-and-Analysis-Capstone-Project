import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('housing_survey.csv')

# Clean the data (assuming cleaning steps are done here)
data.dropna(inplace=True)  # Simple example of dropping missing values

# Define independent and dependent variables
X = data[['size', 'num_rooms', 'other_features']]  # Replace with actual feature names
y = data['market_value']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Sample Data Creation (Replace this with your actual dataset loading)
data1 = {
    'size': [1000, 1500, 2000, 2500, 3000],
    'num_rooms': [2, 3, 4, 4, 5],
    'market_value': [200000, 300000, 400000, 450000, 500000]
}

data2 = {
    'size': [1200, 1600, 1800, 2400, 3200],
    'num_rooms': [3, 3, 4, 4, 5],
    'market_value': [220000, 320000, 370000, 460000, 520000]
}

# Create DataFrames
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

# Merging the datasets
data = pd.concat([df1, df2], ignore_index=True)

# Data Cleaning: Remove duplicates and handle missing values (if any)
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)  # Assuming no NaN values in this example

# Define independent and dependent variables
X = data[['size', 'num_rooms']]  # Features
y = data['market_value']          # Target variable

# Splitting the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and fitting the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Predicted Market Values:", y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Output the model coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)


