# NYC-taxi-trip-duration-prediction
## Project Overview
- A Kaggle competition that require a model which predict trip duration of a taxi in New York City, the collected data for this problem are geo-coordinate, pickup date & time, vendor id, passenger count, and store-and-fwd-flag.

## Data Exploration Analysis
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

## Modeling
---

### Data Pipeline
- spliting features to categorical and numerical
- OneHotEncoding for categorical features
- Quantile and standard for numeric features
- Log transformation for target feature
- Training data using Ridge model with alpha = 1.0

### Evaluation Results
- RMSE (validation ) = 0.48
- R2 (validation) = 0.64
- Mean R2 (cross validation ) = 0.64
- Std R2 (cross validation ) = 0.002

