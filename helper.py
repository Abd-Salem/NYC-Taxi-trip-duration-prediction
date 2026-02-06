MODEL_FILE_PATH = 'trained_model_data.pkl'

def load_model_data():
    '''extracting trained model data for testing'''
    import joblib

    model_data = joblib.load(MODEL_FILE_PATH)  # load model data
    model = model_data['model']         # get pipeline
    input_features = model_data['input_features_names']     # get input features
    return model, input_features


def model_evaluation(model, x, t):
    '''model evaluation using rmse & r2 score'''
    import numpy as np
    from sklearn.metrics import r2_score, mean_squared_error

    preds = model.predict(x)
    mse = mean_squared_error(t, preds)
    score = r2_score(t, preds)

    return np.sqrt(mse), score

