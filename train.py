import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, QuantileTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from helper import load_data, model_evaluation
from prepare_data import prepare_data

RANDOM_STATE = 20
np.random.seed(RANDOM_STATE)
TARGET_FEATURE = 'log_trip_duration'




def train_ridge_model():
    train_df, val_df = load_data('split/train.csv',       # load data
                                 'split/val.csv')

    # prepare data before encoding
    prepare_data(train_df)
    prepare_data(val_df)

    # separate columns according to what will happen to each of them
    features_to_encode = ['vendor_id', 'passenger_count','quarter', 'month', 'dayofweek',
                            'hour','store_and_fwd_flag']
    continuous_features = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
                                   'dropoff_latitude', 'haversine_distance','direction_deg']
    other_features = ['winter', 'spring', 'summer', 'autumn']

    train_features = features_to_encode + continuous_features + other_features

    # one hot encoding, quantile & standard transformation for features
    col_trans = ColumnTransformer([
        ('encoder', OneHotEncoder(handle_unknown='ignore'), features_to_encode),
        ('quantile_trans', QuantileTransformer(output_distribution='normal'), continuous_features),
        ('scaler', StandardScaler(), continuous_features)
    ], remainder='drop')

    # pipeline of training (transformations, model algorithm)
    pipeline = Pipeline(steps=[("col_trans", col_trans), ("model", Ridge(alpha=1, random_state=RANDOM_STATE))])
    model = pipeline.fit(train_df[train_features], train_df[TARGET_FEATURE])

    # evaluate the model using metrics RMSE & r2_score
    rmse_train, score_train = model_evaluation(model, train_df[train_features], train_df[TARGET_FEATURE])
    rmse_val, score_val = model_evaluation(model, val_df[train_features], val_df[TARGET_FEATURE])

    print(f'RMSE (train): {rmse_train:.2f}   -   R2 Score (train): {score_train:.2f}')
    print(f'RMSE (val): {rmse_val:.2f}    -   R2 Score (val): {score_val:.2f}')


    # Cross Validation API shows scores & variance of different models

    # kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    # scores = cross_val_score(pipeline, train_df[train_features], train_df[TARGET_FEATURE],
    #                         cv=kf, scoring='r2')
    # print(f'Scores: {scores}')
    # print(f'Mean: {scores.mean():.2f}   -   Std: {scores.std():.3f}')


if __name__ == '__main__':
    # RMSE(train)=0.48,         RMSE(val)=0.48
    # r2_score(train)=0.64      r2_score(val)=0.64
    train_ridge_model()