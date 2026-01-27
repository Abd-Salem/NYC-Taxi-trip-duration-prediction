import joblib
from helper import load_prepare_data, model_evaluation
from train import TARGET_FEATURE

# files' paths
TEST_FILE_PATH = 'split/test.csv'
MODEL_FILE_PATH = 'trained_model_data.pkl'


def load_model_data():
    '''extracting trained model data for testing'''

    model_data = joblib.load(MODEL_FILE_PATH)  # load model data
    model = model_data['model']         # get pipeline
    input_features = model_data['input_features_names']     # get input features

    return model, input_features

def model_testing():
    '''test model and return results'''

    test_df = load_prepare_data(TEST_FILE_PATH) # load test data
    model, input_features = load_model_data()   # get model and input features
    test_rmse, test_score = model_evaluation(   # evaluate model on test dataset
        model,
        test_df[input_features],
        test_df[TARGET_FEATURE]
    )
    return test_rmse, test_score

if __name__ == '__main__':

    test_rmse, test_score = model_testing()     # get results
    print(f'Test-RMSE: {test_rmse:.2f}  -   Test-R2-score: {test_score:.2f}')
    ########################################
    # Test-RMSE: 0.47 - Test-R2-score: 0.65
    ########################################

