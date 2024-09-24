import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Read the dataset
uber = pd.read_csv("uber-raw-data-sep14.csv")

# Strip column names of extra spaces (if any)
uber.columns = uber.columns.str.strip()

# Convert "Date/Time" column from string data type into DateTime
uber['Date/Time'] = pd.to_datetime(uber['Date/Time'])

# Break down the "Date/Time" column into "Day", "Hour", and "Weekday"
uber['Day'] = uber['Date/Time'].dt.day
uber['Hour'] = uber['Date/Time'].dt.hour
uber['Weekday'] = uber['Date/Time'].dt.weekday

# Prepare data for Random Forest
X = uber[['Hour', 'Day', 'Weekday']]
uber['Trip_Count'] = uber.groupby(['Hour', 'Day', 'Weekday'])['Hour'].transform('count') # or use your trip count if available
y = uber['Trip_Count']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model using Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Random Forest Regressor Mean Squared Error: {mse}")

# Visualize Actual vs Predicted Number of Trips
plt.figure(figsize=(10,6))
plt.scatter(X_test['Hour'], y_test, color='blue', label='Actual Trips')
plt.scatter(X_test['Hour'], y_pred, color='red', label='Predicted Trips')
plt.xlabel('Hour')
plt.ylabel('Number of Trips')
plt.title('Actual vs Predicted Number of Trips per Hour (Random Forest)')
plt.legend()
plt.show()
