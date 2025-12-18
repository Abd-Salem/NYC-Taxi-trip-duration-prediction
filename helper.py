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

def preprocess_data(data, preprocessing_option=1):
    '''
    preprocessing on data according to option:
    1 for MinMaxScaler()
    2 for standardScaler()
    None for other options
    '''
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    processor = None
    if preprocessing_option == 1:
        processor = MinMaxScaler()
        return processor, processor.fit_transform(data)
    elif preprocessing_option == 2:
        processor = StandardScaler()
        return processor, processor.fit_transform(data)

    return processor, data


def model_evaluation(model, x, t):
    '''model evaluation using rmse & r2 score'''
    from sklearn.metrics import r2_score, mean_squared_error

    preds = model.predict(x)
    rmse = mean_squared_error(t, preds, squared=False)
    score = r2_score(t, preds)

    return rmse, score


