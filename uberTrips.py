import pandas as pd 
import matplotlib.pyplot as plt 

# Read the dataset
uber = pd.read_csv("uber-raw-data-sep14.csv")

# Strip column names of extra spaces (if any)
uber.columns = uber.columns.str.strip()

# Display the first few rows and dataset information
print(uber.head())
uber.info()

# Convert "Date/Time" column from string data type into DateTime
uber['Date/Time'] = pd.to_datetime(uber['Date/Time'])

# Break down the "Date/Time" column into "Day", "Hour", and "Weekday"
uber['Day'] = uber['Date/Time'].dt.day
uber['Hour'] = uber['Date/Time'].dt.hour
uber['Weekday'] = uber['Date/Time'].dt.weekday

# Check for missing or invalid values in Longitude and Latitude
print(uber[['Lon', 'Lat']].isnull().sum())  # Check for missing values

# Plot: Density of Trips per Day
fig, ax = plt.subplots(figsize=(12, 6))
plt.hist(uber.Day, width=0.6, bins=30, color='#1f77b4', edgecolor='black', alpha=0.7)
plt.title("Density of Trips per Day", fontsize=18, fontweight='bold', color='#333333')
plt.xlabel("Day", fontsize=14, fontweight='bold')
plt.ylabel("Density of Rides", fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Plot: Density of Trips per Weekday
fig, ax = plt.subplots(figsize=(12, 6))
plt.hist(uber.Weekday, width=0.6, range=(0, 6.5), bins=7, color='#2ca02c', edgecolor='black', alpha=0.7)
plt.title("Density of Trips per Weekday", fontsize=18, fontweight='bold', color='#333333')
plt.xlabel("Weekday", fontsize=14, fontweight='bold')
plt.ylabel("Density of Rides", fontsize=14, fontweight='bold')
plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], fontsize=12)
plt.yticks(fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Plot: Density of Trips per Hour
fig, ax = plt.subplots(figsize=(12, 6))
plt.hist(uber.Hour, width=0.6, bins=24, color='#ff7f0e', edgecolor='black', alpha=0.7)
plt.title("Density of Trips per Hour", fontsize=18, fontweight='bold', color='#333333')
plt.xlabel("Hour", fontsize=14, fontweight='bold')
plt.ylabel("Density of Rides", fontsize=14, fontweight='bold')
plt.xticks(ticks=range(0, 24), fontsize=12)
plt.yticks(fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Plot: Distribution of Trip Locations (Longitude vs Latitude)
fig, ax = plt.subplots(figsize=(12, 6))
x = uber.Lon
y = uber.Lat
plt.scatter(x, y, color='#9467bd', edgecolor='black', alpha=0.6, s=50)
plt.title("Distribution of Trip Locations", fontsize=18, fontweight='bold', color='#333333')
plt.xlabel("Longitude", fontsize=14, fontweight='bold')
plt.ylabel("Latitude", fontsize=14, fontweight='bold')

# Set x and y axis limits for better geographical accuracy
plt.xlim([-74.2, -73.7])  # New York region
plt.ylim([40.5, 41])      # New York region
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
