from helper import load_model_data, model_evaluation
from train import TARGET_FEATURE
from prepare_data import prepare_data
import numpy as np
import pandas as pd

# files' paths
TEST_FILE_PATH = 'split/test.csv'

def model_testing():
    '''test model and return results'''

    test_df = pd.read_csv(TEST_FILE_PATH)
    processed_test_df = prepare_data(test_df) # load test data
    processed_test_df[f'log_{TARGET_FEATURE}'] = np.log1p(test_df[TARGET_FEATURE])
    model, input_features = load_model_data()   # get model and input features
    test_rmse, test_score = model_evaluation(   # evaluate model on test dataset
        model,
        processed_test_df[input_features],
        processed_test_df[f'log_{TARGET_FEATURE}']
    )
    return test_rmse, test_score

if __name__ == '__main__':

    test_rmse, test_score = model_testing()     # get results
    print(f'Test-RMSE: {test_rmse:.2f}  -   Test-R2-score: {test_score:.2f}')
    #########################################
    # Test-RMSE: 0.47 - Test-R2-score: 0.65 #
    #########################################

