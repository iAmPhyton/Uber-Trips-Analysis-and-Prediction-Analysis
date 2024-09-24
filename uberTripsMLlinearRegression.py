import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 

# Read the dataset
uber = pd.read_csv("uber-raw-data-sep14.csv")

#correctly parsing the date/time column to datetime format
uber['Date/Time'] = pd.to_datetime(uber['Date/Time'])
uber 

#creating 'Day', 'Hour', and 'Weekday' columns by extracting from 'Date/Time'
uber['Day'] = uber['Date/Time'].dt.day
uber['Hour'] = uber['Date/Time'].dt.hour
uber['Weekday'] = uber['Date/Time'].dt.weekday
uber 


#preparing the dataset by counting the number of trips per hour
uber['Trip_Count'] = uber.groupby(['Hour'])['Hour'].transform('count')
uber 
#This line of code above will create a new column Trip_Count that stores the number of trips per hour 

#Feature Selection
#using hour, day, weekday as the features for prediciting the trip count
#features (Hour, Day, Weekday) and target (Trip_Count)
X = uber[['Hour', 'Day', 'Weekday']]
y = uber['Trip_Count']

#Train-Test Split
from sklearn.model_selection import train_test_split

#spliting the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#model training
from sklearn.linear_model import LinearRegression
#creating and training the model (Linear Regression)
model = LinearRegression()
model.fit(X_train, y_train)

#making predictions on the test set
y_pred = model.predict(X_test)

#evaluating the model
from sklearn.metrics import mean_squared_error

# Evaluate the model using Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

#visualizing the final results
import matplotlib.pyplot as plt
#plotting actual vs predicted trips
plt.figure(figsize=(10,6))
plt.scatter(X_test['Hour'], y_test, color='blue', label='Actual Trips')
plt.scatter(X_test['Hour'], y_pred, color='red', label='Predicted Trips')
plt.xlabel('Hour')
plt.ylabel('Number of Trips')
plt.title('Actual vs Predicted Number of Trips per Hour')
plt.legend()
plt.show() 