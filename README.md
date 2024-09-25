Uber Trip Analysis and Prediction

Overview: 

This project analyzes Uber trip data for September 2014 and builds machine learning models to predict the number of trips based on features like hour, day, and weekday. The project also includes visualizations to explore the patterns in the trip data and compares the performance of two predictive models: Linear Regression and Random Forest Regression.

Dataset:

The dataset used in this project contains raw Uber trip data for September 2014, including the following columns:
- `Date/Time`: The date and time of the trip.
- `Lat`: Latitude of the trip's starting location.
- `Lon`: Longitude of the trip's starting location.

The dataset is also transformed by extracting the following:
- `Day`: Day of the month.
- `Hour`: Hour of the day.
- `Weekday`: Day of the week.

The above features predict the 'Trip Count': — the number of trips taken at each hour.

Project Structure:

- Data Preprocessing: The dataset is cleaned and features: `Day`, `Hour`, and `Weekday` are extracted from the `Date/Time` column.
  
- Visualizations: 
  - Distribution of trips by day, weekday, and hour.
  - Scatter plot of trip locations (latitude and longitude).
  
- Machine Learning Models:
  - Linear Regression: Used to predict trip count based on the extracted features.
  - Random Forest Regressor: Another model used to predict trip count, offering improved performance over Linear Regression.
  
- Model Comparison: Visual and quantitative comparison of the two models using Mean Squared Error (MSE) and scatter plots of actual vs. predicted trip counts.

Dependencies:

To run the project, you need the following Python libraries:
- `pandas`
- `matplotlib`
- `scikit-learn`

You can install the required packages by running the below code via the cmd prompt on your computer:
pip install pandas matplotlib scikit-learn

Code Summary:

1. Data Loading and Preprocessing:
   - Convert `Date/Time` to datetime format.
   - Extract `Day`, `Hour`, and `Weekday` features.
   - Create the `Trip_Count` column by grouping trips by hour.

2. Visualization:
   - Create histograms for trip counts per day, hour, and weekday.
   - Scatter plot showing trip locations by latitude and longitude.

3. Model Training and Prediction:
   - Train 'Linear Regression' and 'Random Forest' models using `Hour`, `Day`, and `Weekday` as input features and `Trip_Count` as the target variable.
   - Split the data into training and test sets (80% training, 20% testing).
   - Evaluate both models using **Mean Squared Error (MSE)**.

4. Results and Comparison:
   - The 'Random Forest' model performed better than 'Linear Regression' based on the MSE.
   - Visualize the actual vs. predicted number of trips for both models.

## How to Run

1. Clone the repository:
   
   git clone <your-repo-url>
   cd <your-repo-directory>

3. Install dependencies:
   
   pip install -r requirements.txt

4. Run the Python script:
   
   python uber_trip_analysis.py

Results:

- Linear Regression: Basic regression model that offers a quick insight into the data but with higher prediction errors.
- Random Forest Regressor: A learning model that significantly improves the prediction accuracy for Uber trip counts.

Based on the Mean Squared Error (MSE):
- Linear Regression MSE: 301561.5642123331
- Random Forest Regressor MSE: 0.0

Conclusion:

The Random Forest Regressor outperforms the Linear Regression model in predicting the number of trips per hour. This project demonstrates how different machine learning models can be applied to real-world data to gain insights and make predictions.
