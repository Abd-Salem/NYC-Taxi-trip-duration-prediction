
def load_data(train_file_path:str='split/train.csv', val_file_path:str='split/val.csv'):
    '''
    :param train_file_path:
    :param val_file_path:
    :return: train & val data
    '''
    import pandas as pd

    train_df = pd.read_csv(train_file_path)
    val_df = pd.read_csv(val_file_path)

    return train_df, val_df


def model_evaluation(model, x, t):
    '''model evaluation using rmse & r2 score'''
    import numpy as np
    from sklearn.metrics import r2_score, mean_squared_error

    preds = model.predict(x)
    mse = mean_squared_error(t, preds)
    score = r2_score(t, preds)

    return np.sqrt(mse), score


