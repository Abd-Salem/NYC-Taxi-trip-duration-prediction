
def haversine_distance_and_direction(pick_lat, pick_long, drop_lat, drop_long):
    """Calculate distance(km) and direction degree using vectorized operations"""
    import numpy as np

    radius = 6371  # Radius of Earth (km)
    pick_lat_rad, pick_long_rad = np.radians(pick_lat), np.radians(pick_long)
    drop_lat_rad, drop_long_rad = np.radians(drop_lat), np.radians(drop_long)

    dlat = drop_lat_rad - pick_lat_rad
    dlong = drop_long_rad - pick_long_rad

    # calculate distance for all points
    a = np.sin(dlat / 2) ** 2 + np.cos(pick_lat_rad) * np.cos(drop_lat_rad) * np.sin(dlong / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    distances = radius * c

    # Calculate bearing for all points
    x = np.sin(dlong) * np.cos(drop_lat_rad)
    y = np.cos(pick_lat_rad) * np.sin(drop_lat_rad) - np.sin(pick_lat_rad) * np.cos(drop_lat_rad) * np.cos(dlong)

    bearings_rad = np.arctan2(x, y)
    bearings_deg = np.degrees(bearings_rad)
    bearings_deg = (bearings_deg + 360) % 360

    return distances, bearings_deg




def prepare_data(df):
    '''prepare data before feature engineering step'''
    import numpy as np

    # extract features from data & time column
    df['pickup_datetime'] = df['pickup_datetime'].astype('datetime64[ns]')
    df['month'] = df['pickup_datetime'].dt.month
    df['quarter'] = df['pickup_datetime'].dt.quarter
    df['dayofweek'] = df['pickup_datetime'].dt.dayofweek
    df['dayofyear'] = df['pickup_datetime'].dt.dayofyear
    df['hour'] = df['pickup_datetime'].dt.hour

    # extract season features
    df['winter'] = df['month'].isin([12, 1, 2]).astype(float)
    df['spring'] = df['month'].isin([3, 4, 5]).astype(float)
    df['summer'] = df['month'].isin([6, 7, 8]).astype(float)
    df['autumn'] = df['month'].isin([9, 8, 10]).astype(float)

    # log transformation for target feature
    df['log_trip_duration'] = np.log1p(df['trip_duration'])

    # calculate haversine distance and direction degree
    df['haversine_distance'], df['direction_deg'] = haversine_distance_and_direction(df['pickup_latitude'],
                                                  df['pickup_longitude'],df['dropoff_latitude'],df['dropoff_longitude'])

    # extract 8 half quarters for different degrees
    df['direction_deg'] = df['direction_deg'] % 360  # 360 = 0
    df.loc[:, '(1st/8)'] = ((df['direction_deg'] >= 0.0) & (df['direction_deg'] <= 45.0)).astype(float)
    df.loc[:, '(2nd/8)'] = ((df['direction_deg'] > 45.0) & (df['direction_deg'] <= 90.0)).astype(float)
    df.loc[:, '(3rd/8)'] = ((df['direction_deg'] > 90.0) & (df['direction_deg'] <= 135.0)).astype(float)
    df.loc[:, '(4th/8)'] = ((df['direction_deg'] > 135.0) & (df['direction_deg'] <= 180.0)).astype(float)
    df.loc[:, '(5th/8)'] = ((df['direction_deg'] > 180.0) & (df['direction_deg'] <= 225.0)).astype(float)
    df.loc[:, '(6th/8)'] = ((df['direction_deg'] > 225.0) & (df['direction_deg'] <= 270.0)).astype(float)
    df.loc[:, '(7th/8)'] = ((df['direction_deg'] > 270.0) & (df['direction_deg'] <= 315.0)).astype(float)
    df.loc[:, '(8th/8)'] = ((df['direction_deg'] > 315.0) & (df['direction_deg'] <= 360.0)).astype(float)

    df.drop(columns=['id','pickup_datetime', 'trip_duration'], inplace=True)  # drops