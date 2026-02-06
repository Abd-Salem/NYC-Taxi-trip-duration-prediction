import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, QuantileTransformer, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from helper import model_evaluation
import joblib

RANDOM_STATE = 20
np.random.seed(RANDOM_STATE)
TARGET_FEATURE = 'trip_duration'



def data_preprocessing_pipeline(use_poly_trans=False, degree=2):
    '''data preprocessing steps before training model'''

    # separate columns according to what will happen to each of them
    categorical_features = ['vendor_id', 'passenger_count','quarter', 'month', 'dayofweek',
                            'hour','store_and_fwd_flag']
    continuous_features = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
                                   'dropoff_latitude', 'haversine_distance','direction_deg']
    other_features = ['winter', 'spring', 'summer', 'autumn', 'rush_hour', 'rush_day', '(1st/8)',
                      '(2nd/8)', '(3rd/8)','(4th/8)', '(5th/8)','(6th/8)', '(7th/8)', '(8th/8)']

    input_features = categorical_features + continuous_features + other_features

    # conditional expression for the transformer
    transformer = PolynomialFeatures(degree=degree, include_bias=False) if use_poly_trans \
        else QuantileTransformer(output_distribution='normal')

    # continuous pipeline selected transformer
    continuous_pipeline = Pipeline([
        ('feature_transformer', transformer),
        ('processor', StandardScaler())
    ])

    # all features transformation
    col_trans = ColumnTransformer([
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('continuous', continuous_pipeline, continuous_features)
    ], remainder='passthrough')

    return input_features, col_trans


def train_ridge_model(train_df, input_features, target_feature, col_trans):
    '''training ridge model on our data'''
    # pipeline of training (transformations, model algorithm)
    pipeline = Pipeline(steps=[("col_trans", col_trans), ("model", Ridge(alpha=1, random_state=RANDOM_STATE))])
    model = pipeline.fit(train_df[input_features], train_df[target_feature])
    return model

def cross_validation(pipeline, x, t):
    '''Cross Validation API shows scores & variance of different models'''

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipeline, x, t,
                            cv=kf, scoring='r2')
    print(f'Scores: {scores}')
    print(f'Mean: {scores.mean():.2f}   -   Std: {scores.std():.3f}')


if __name__ == '__main__':

    version = 1
    processed_train_path = f'processed_data/{version}/train.csv'
    processed_val_path = f'processed_data/{version}/val.csv'

    train_df = pd.read_csv(processed_train_path)   # load processed train data
    val_df = pd.read_csv(processed_val_path)       # load processed val data
    input_features, col_trans = data_preprocessing_pipeline()      # data preprocessing pipeline

    # log transformation for target feature
    train_df[f'log_{TARGET_FEATURE}'] = np.log1p(train_df[TARGET_FEATURE])
    val_df[f'log_{TARGET_FEATURE}'] = np.log1p(val_df[TARGET_FEATURE])

    # train model on data after preprocessing it
    model = train_ridge_model(
        train_df,
        input_features,
        f'log_{TARGET_FEATURE}',
        col_trans
    )

    # model evaluation on train data and val data
    # RMSE & r2 score evaluations
    rmse_train, r2_score_train =  model_evaluation(
        model,
        train_df[input_features],
        train_df[f'log_{TARGET_FEATURE}']
    )
    rmse_val, r2_score_val = model_evaluation(
        model,
        val_df[input_features],
        val_df[f'log_{TARGET_FEATURE}']
    )

    # show evaluation results
    print(f'train-RMSE: {rmse_train:.2f},   -   train-R2-score: {r2_score_train:.2f}')
    print(f'val-RMSE: {rmse_val:.2f},   -   val-R2-score: {r2_score_val:.2f}')
    ################################################
    # train-RMSE: 0.47, -   train-R2- score: 0.65
    # val-RMSE: 0.47,   -   val-R2-score: 0.65
    #################################################

    # save model and it's metadata as .pkl file
    model_data = {
        'model': model,
        'data_path': f'processed_data/{version}',
        'RMSE_train': rmse_train,
        'r2_score_train': r2_score_train,
        'RMSE_val': rmse_val,
        'r2_score_val': rmse_val,
        'input_features_names': input_features,
        'data_processor_pipeline': col_trans,
        'random_state': RANDOM_STATE
    }

    # save model data
    joblib.dump(model_data, 'trained_model_data.pkl')