import pandas as pd

from utils.model_utils import Feature
from data.process import DataProcessor, DataSet
from pre_game.pipeline.data_processer import XTrainConstructor, XTestConstructor
from pipeline.pre_processer import DataEncoder, TargetMapper, DataAligner

import xgboost as xgb
from sklearn.svm import SVC
from catboost import CatBoostClassifier

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def get_feature_params():
    """
    Returns a dictionary of feature flags, enabling or disabling certain features for the model.
    """
    return {
        Feature.GOAL_STATS.value: True,
        Feature.SHOOTING_STATS.value: True,
        Feature.POSSESSION_STATS.value: True,
        Feature.ODDS.value: True,
        Feature.XG.value: True,
        Feature.HOME_AWAY_RESULTS.value: True,
        Feature.CONCEDED_STATS.value: True,
        Feature.LAST_N_MATCHES.value: True,
        Feature.WIN_STREAK.value: True,
    }


def train_model(model, model_name, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series = None):
    """
    Trains a given model and evaluates its performance using various metrics such as accuracy, F1 score, precision, and recall.
    
    Parameters:
    model: The machine learning model to train.
    model_name: Name of the model (used for logging).
    X_train: Training feature set.
    y_train: Training labels.
    X_test: Test feature set.
    y_test: Test labels.

    Returns:
    Series: Predicted labels for the test set.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Perform k-fold cross-validation
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold)

    print(f'{model_name}:')
    print(f'Predictions: {y_pred}')
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('F1 Score:', f1_score(y_test, y_pred, average='macro'))
    print('Precision:', precision_score(y_test, y_pred, average='macro'))
    print('Recall:', recall_score(y_test, y_pred, average='macro'))
    print(f'Cross Validation Accuracy: mean={round(cv_results.mean(), 5)}, std={round(cv_results.std(), 5)}\n')

    return y_pred


def pre_process_data(raw_df) -> tuple[DataSet, DataSet, list[str]]:
    """
    Pre-processes the raw data by splitting it into training and test sets and retrieving the list of unique teams.

    Parameters:
    raw_df: The raw DataFrame containing the data to be processed.

    Returns:
    tuple: A tuple containing the training dataset, test dataset, and the list of unique teams.
    """
    data_processor = DataProcessor(raw_df)
    unique_teams = data_processor.get_unique_teams()
    train_data, test_data = data_processor.split_data(train_test_ratio=0.95)

    return train_data, test_data, unique_teams


def feature_engineering(train: DataSet, test: DataSet, teams, feature_flags):
    """
    Performs feature engineering on the training and test datasets, including constructing feature tables,
    encoding the data, and aligning columns.

    Parameters:
    train: Training dataset.
    test: Test dataset.
    teams: List of unique teams.
    feature_flags: Dictionary of feature flags.

    Returns:
    tuple: The engineered feature sets and labels for both training and test data.
    """
    X_train = XTrainConstructor(train.X, teams, **feature_flags).construct_table()
    X_train = DataEncoder(X_train).process()
    y_train = TargetMapper(train.y).encode_labels()

    X_test = XTestConstructor(test.X, train.X, teams, **feature_flags).construct_table()
    X_test = DataEncoder(X_test).process()
    y_test = TargetMapper(test.y).encode_labels()

    X_train, X_test = DataAligner(X_train, X_test).align_columns()

    return X_train, y_train, X_test, y_test


def run(raw_df, feature_flags=get_feature_params()):
    """
    Runs the entire pipeline from data pre-processing to training multiple machine learning models.

    Parameters:
    raw_df: The raw DataFrame to process and use for model training.
    feature_flags: Dictionary of feature flags for feature engineering.
    """
    # Pre-process the data
    train_data, test_data, unique_teams = pre_process_data(raw_df)
    
    # Perform feature engineering
    X_train, y_train, X_test, y_test = feature_engineering(train_data, test_data, unique_teams, feature_flags)

    # Train and evaluate various models
    lr_model = LogisticRegression(max_iter=1000)
    train_model(lr_model, 'Logistic Regression', X_train, y_train, X_test, y_test)

    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    train_model(rf_model, 'Random Forest', X_train, y_train, X_test, y_test)

    xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=3)
    train_model(xgb_model, 'XGBoost', X_train, y_train, X_test, y_test)

    catboost_model = CatBoostClassifier(iterations=1000, depth=5, learning_rate=0.1, loss_function='MultiClass')
    catboost_model.set_params(logging_level='Silent')
    train_model(catboost_model, 'CatBoost', X_train, y_train, X_test, y_test)

    svm_model = SVC(kernel='linear')
    train_model(svm_model, 'SVM', X_train, y_train, X_test, y_test)

    mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
    train_model(mlp_model, 'MLP Classifier', X_train, y_train, X_test, y_test)

    adb_model = AdaBoostClassifier(n_estimators=100)
    train_model(adb_model, 'AdaBoost', X_train, y_train, X_test, y_test)
