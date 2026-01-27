
def load_data(file_path):
    '''
    :param file_path:
    :return: loaded data
    '''
    import pandas as pd

    loaded_data = pd.read_csv(file_path)

    return loaded_data


def model_evaluation(model, x, t):
    '''model evaluation using rmse & r2 score'''
    import numpy as np
    from sklearn.metrics import r2_score, mean_squared_error

    preds = model.predict(x)
    mse = mean_squared_error(t, preds)
    score = r2_score(t, preds)

    return np.sqrt(mse), score


