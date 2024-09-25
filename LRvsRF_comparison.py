import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

#loading the dataset
uber = pd.read_csv("uber-raw-data-sep14.csv") 
uber 

#parsing the 'Date/Time column to datetime format
uber['Date/Time'] = pd.to_datetime(uber['Date/Time'])

#breaking down 'Date/Time column into 'Day', 'Hour', and 'Weekday'
uber['Day'] =uber['Date/Time'].dt.day
uber['Hour'] =uber['Date/Time'].dt.hour
uber['Weekday'] =uber['Date/Time'].dt.weekday

#preparing the dataset by counting the number of trips per hour/day/weekday
uber['Trip_Count'] = uber.groupby(['Hour', 'Day', 'Weekday'])['Hour'].transform('count')

#features (Hour, day, Weekday) and target (Trip_Count)
x = uber[['Hour', 'Day', 'Weekday']]
y = uber['Trip_Count']

#train-test split (80% training, 20% testing)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

#Liner Regression Model
lr_model = LinearRegression()
lr_model.fit(x_train, y_train)
y_pred_lr = lr_model.predict(x_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
print(f"Linear Regression Mean Squared Error: {mse_lr}")

#visualising Actual vs Predicted for Linear Regression
plt.figure(figsize=(10,6))
plt.scatter(x_test['Hour'], y_test, color='blue', label='Actual Trips')
plt.scatter(x_test['Hour'], y_pred_lr, color='red', label='Predicted Trips (Linear Regression)', alpha=0.7)
plt.xlabel('Hour')
plt.ylabel('Number of Trips')
plt.title('Actual vs Predicted Number of Trips per Hour (Linear Regression)')
plt.legend()
plt.show() 

#Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)
y_pred_rf = rf_model.predict(x_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
print(f"Random Forest Mean Square Error: {mse_rf}")

#visuaising Actual vs Predicted for Random Forest
plt.figure(figsize=(10,6))
plt.scatter(x_test['Hour'], y_test, color='blue', label='Actual Trips')
plt.scatter(x_test['Hour'], y_pred_rf, color='green', label='Predicted Trips (Random Forest)', alpha=0.7)
plt.xlabel('Hour')
plt.ylabel('Number of Trips')
plt.title('Actual vs Predicted Number of Trips per Hour (Random Forest)')
plt.legend()
plt.show()

#visualisng comparison: Linear Regression vs Random Forest
plt.figure(figsize=(10,6))
plt.scatter(x_test['Hour'], y_test, color='blue', label='Actual Trips', alpha=0.5)
plt.scatter(x_test['Hour'], y_pred_lr, color='red', label='Predicted Trips (Linear Regression)', alpha=0.7)
plt.scatter(x_test['Hour'], y_pred_rf, color='green', label='Predicted Trips (Random Forest)', alpha=0.7)
plt.xlabel('Hour')
plt.ylabel('Number of Trips')
plt.title('Comparison of Predicted Trips (Linear Regression vs Random Forest)')
plt.legend()
plt.show()
