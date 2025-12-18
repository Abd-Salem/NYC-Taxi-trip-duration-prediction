def prepare_data(df):
    import numpy as np

    df.drop(columns=['id'], inpace=True)
    df['pickup_datetime'] = df['pickup_datetime'].astype('datetime64[ns]')
    df['month'] = df['pickup_datetime'].dt.month
    df['quarter'] = df['pickup_datetime'].dt.quarter
    df['dayofweek'] = df['pickup_datetime'].dt.quarter
    df['hour'] = df['pickup_datetime'].dt.quarter
    df['log_trip_duration'] = np.log1p(df['trip_duration'])

    df['haversine_distance'] = haversine_distance(df['pickup_latitude'],
                                                  df['pickup_longitude'],
                                                  df['dropoff_latitude'],
                                                  df['dropoff_longitude'])


def haversine_distance(pick_lat, pick_long, drop_lat, drop_long):
    """Calculate distance in km using vectorized operations"""
    import numpy as np

    radius = 6371  # Radius of Earth (km)
    pick_lat_rad, pick_long_rad = np.radians(pick_lat), np.radians(pick_long)
    drop_lat_rad, drop_long_rad = np.radians(drop_lat), np.radians(drop_long)

    dlat = drop_lat_rad - pick_lat_rad
    dlong = drop_long_rad - pick_long_rad

    a = np.sin(dlat / 2) ** 2 + np.cos(pick_lat_rad) * np.cos(drop_lat_rad) * np.sin(dlong / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return radius * c


def calculate_bearings_batch(pick_lat, pick_long, drop_lat, drop_long):
    """Calculate bearings for arrays of points."""
    import numpy as np
    # Convert to radians
    pick_lat_rad = np.radians(pick_lat)
    pick_long_rad = np.radians(pick_long)
    drop_lat_rad = np.radians(drop_lat)
    drop_long_rad = np.radians(drop_long)

    dlon = drop_long_rad - pick_long_rad

    # Calculate bearing for all points
    x = np.sin(dlon) * np.cos(drop_lat_rad)
    y = np.cos(pick_lat_rad) * np.sin(drop_lat_rad) - np.sin(pick_lat_rad) * np.cos(drop_lat_rad) * np.cos(dlon)

    bearings_rad = np.arctan2(x, y)
    bearings_deg = np.degrees(bearings_rad)
    bearings_deg = (bearings_deg + 360) % 360

    return bearings_deg