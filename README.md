# ***NYC-taxi-trip-duration-prediction***

> ## Project Overview
---
- Predict the trip duration for taxi rides in New York City based on pickup/dropoff locations, temporal features, and trip metadata. This is a Kaggle competition focused on regression modeling.
- Data Source: https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data

> ## Project Objectives
---
- Build a machine learning model to accurately predict taxi trip duration
- Extract meaningful features from geo-coordinates and datetime data
- Handle skewed distributions and outliers in the target variable
- Create a reproducible ML pipeline from data preparation to model deployment

> ## Dataset Overview
---
### Input Features:

- Geographical: Pickup/dropoff longitude and latitude
- Temporal: Pickup date and time
- Metadata: Vendor ID, passenger count, store-and-forward flag

### Target Variable:

- trip_duration: Duration of the trip in seconds

> ## Dependencies
---
    pip3 install -r requirements.txt
> ## Project Structure
---
    NYC-Taxi-trip-duration-prediction/
    │
    ├── docs/                           # Documentation folder
    │
    ├── processed_data/
    │   └── 1/                          # Processed data files version 1
    │   └── 2/	         # processed data files version 2
    │	 
    ├── split/                          # Train/test split data
    │
    ├── .gitattributes                  # Git attributes configuration
    ├── .gitignore                      # Git ignore rules
    │
    ├── EDA-trip-duration.ipynb         # Exploratory Data Analysis notebook
    ├── README.md                       # Project documentation
    │
    ├── helper.py                       # Helper functions
    ├── load_test.py                    # Testing/loading script
    ├── prepare_data.py                 # Data preparation pipeline
    ├── train.py                        # Model training script
    │
    ├── requirements.txt                # Python dependencies
    └── trained_model_data.pkl          # Saved trained model
> ## Key Findings from EDA
---
### Trip-Duration (target feature)
- Feature data is skewed
- Feature data has normal distribution when applying log transformation
- Feature data has some outliers
### Vendor ID
- ID of the vehicle vendor (1, 2)
- Average of the trip durations of vendor(2) is longer than vendor(1)
### Passenger Count
- Number of passengers in the trip, number is assigned by the driver
- After removing outliers from target feature passenger count (7) is removed
- passenger count (0) may be for a specific purpose so, this may not help the model
### Geographical features
- Pickup longitude & Pickup latitude (pickup point)
- Drop off longitude & Drop off latitude (drop off point)
- Some (pickup-Dropoff) points are outside New York City
- Haversine distance is calculated for trips and showed that most of trips cover 1km to 23km
- Speed is also calculated and ranged from 10 km/h to 40 km/h over different times
### Date time features
- Only records for months 1,2,3,4,5,6 in 2016
- This shows low diversity in the data
- Longer trips are in summer, normal days (not weekend) with low speed (crowd streets)
- Hours from 12pm to 5pm shows longer durations with low speed (crowd streets)

> ## Modeling
---
### Data Pipeline
    1. Feature Splitting → Categorical & Numerical
    2. Categorical Features → OneHotEncoding
    3. Numerical Features → QuantileTransformer + StandardScaler
    4. Target Variable → Log Transformation (np.log1p)
### Model Selection:
- Algorithm: Ridge Regression (L2 regularization)

### Evaluation Results
- RMSE (validation ) = 0.47
- R2 (validation) = 0.65
- Mean R2 (Cross-Validation ) = 0.65
- Std R2 (Cross-Validation ) = 0.005

>## Future Improvements
---

1- Advanced Models: Try Gradient Boosting (XGBoost)
2- Feature Engineering: Add weather data, traffic patterns, holidays
3- Hyperparameter Tuning: Grid/Random search for optimal parameters
4- Ensemble Methods: Combine multiple models for better predictions
5- Real-time Prediction: Deploy model as API service

