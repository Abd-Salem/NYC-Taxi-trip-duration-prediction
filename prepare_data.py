
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



def extract_time_features(df):
    """extract time features"""

    # extract features from data & time column
    df['pickup_datetime'] = df['pickup_datetime'].astype('datetime64[ns]')
    df['month'] = df['pickup_datetime'].dt.month
    df['quarter'] = df['pickup_datetime'].dt.quarter
    df['dayofweek'] = df['pickup_datetime'].dt.dayofweek
    df['hour'] = df['pickup_datetime'].dt.hour
    df['rush_hour'] = df['hour'].between(12,18).astype(float)
    df['rush_day'] = df['dayofweek'].between(2,4).astype(float)

    # extract season features
    df['winter'] = ((df['month'] == 12) | (df['month'] <= 2)).astype(float)
    df['spring'] = df['month'].between(3, 5).astype(float)
    df['summer'] = df['month'].between(6, 8).astype(float)
    df['autumn'] = df['month'].between(9, 10).astype(float)


def extract_direction_degree_features(df, direction_degrees_feature_name):
    """calculate direction of the trip"""
    # extract 8 half quarters for different degrees
    df['direction_deg'] = df[direction_degrees_feature_name] % 360  # 360 = 0
    df.loc[:, '(1st/8)'] = ((df[direction_degrees_feature_name] >= 0.0) & (df[direction_degrees_feature_name] <= 45.0)).astype(float)
    df.loc[:, '(2nd/8)'] = ((df[direction_degrees_feature_name] > 45.0) & (df[direction_degrees_feature_name] <= 90.0)).astype(float)
    df.loc[:, '(3rd/8)'] = ((df[direction_degrees_feature_name] > 90.0) & (df[direction_degrees_feature_name] <= 135.0)).astype(float)
    df.loc[:, '(4th/8)'] = ((df[direction_degrees_feature_name] > 135.0) & (df[direction_degrees_feature_name] <= 180.0)).astype(float)
    df.loc[:, '(5th/8)'] = ((df[direction_degrees_feature_name] > 180.0) & (df[direction_degrees_feature_name] <= 225.0)).astype(float)
    df.loc[:, '(6th/8)'] = ((df[direction_degrees_feature_name] > 225.0) & (df[direction_degrees_feature_name] <= 270.0)).astype(float)
    df.loc[:, '(7th/8)'] = ((df[direction_degrees_feature_name] > 270.0) & (df[direction_degrees_feature_name] <= 315.0)).astype(float)
    df.loc[:, '(8th/8)'] = ((df[direction_degrees_feature_name] > 315.0) & (df[direction_degrees_feature_name] <= 360.0)).astype(float)

def prepare_data(df):
    '''prepare data before feature engineering step'''
    import numpy as np

    # extract time features ( day, hour, month, quarter, ...)
    extract_time_features(df)

    # calculate haversine distance
    df['haversine_distance'], df['direction_deg'] = (haversine_distance_and_direction(
        df['pickup_latitude'],df['pickup_longitude'],
        df['dropoff_latitude'],df['dropoff_longitude']
    )
    )

    # extract 8 categorical features for different degrees
    extract_direction_degree_features(df, 'direction_deg')

    # drop target feature & useless features
    df.drop(columns=['id','pickup_datetime'], inplace=True)  # drops

    return df


if __name__ == '__main__':
    import pandas as pd
    import os, json

    # data paths
    train_path = 'split/train.csv'
    val_path = 'split/val.csv'

    # load data
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # prepare data
    processed_train = prepare_data(train_df.copy())
    processed_val = prepare_data(val_df.copy())

    version = 1
    processed_data_path = f'processed_data/{version}'
    if not os.path.exists(processed_data_path):     # check if not exist to create one
        os.makedirs(processed_data_path)

    # save processed data (train, val) as .csv file to be ready for training and testing
    processed_train.to_csv(f'{processed_data_path}/train.csv', index=False)
    processed_val.to_csv(f'{processed_data_path}/val.csv', index=True)


    processed_data_metadata = {
        'version' : version,
        'version_description' : 'Extracting & adding time features and direction degree features',
        'features_names' : processed_train.columns.tolist(),
        'train_shape' : processed_train.shape,
        'val_shape' : processed_val.shape
    }

    # save processed data metadata as .json file
    with open(f'{processed_data_path}/metadata.json', 'w') as json_file:
        json.dump(processed_data_metadata, json_file, indent=4)