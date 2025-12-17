# NYC-taxi-trip-duration-prediction
## Project Overview
- A Kaggle competition that require a model which predict trip duration of a taxi in New York City, the collected data for this problem are geo-coordinate, pickup date & time, vendor id, passenger count, and store-and-fwd-flag.

## Data Exploration Analysis
### Trip-Duration (target feature)
- Trip duration in seconds
- Data description shows that there is a skewness.
- Data contains outliers.
### Vendor ID
- ID of the vehicle vendor (1, 2)
### Passenger Count
- Number of passengers in the trip
- Provided by the driver
- There are some trips that have 0 or 7 passengers
### Geographical features
- Pickup longitude & Pickup latitude (pickup point)
- Drop off longitude & Drop off latitude (drop off point)
### Store and forward flag
- Flag that shows the duration is recorded inside the car then forwarded or was connected to the server
- Y: stored and forwarded (disconnected)
- N: not stored and forwarded (connected)
