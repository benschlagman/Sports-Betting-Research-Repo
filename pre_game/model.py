import pandas as pd

from utils.model_utils import Feature
from data.process import DataProcessor, DataSet
from pre_game.pipeline.data_processer import XTrainConstructor, XTestConstructor
from pipeline.pre_processer import DataEncoder, TargetMapper, DataAligner

import xgboost as xgb
from sklearn.svm import SVC
from catboost import CatBoostClassifier

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier


def get_feature_params():
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


def train_model(model, name, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series = None):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # k-fold cross validation
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    results = cross_val_score(model, X_train, y_train, cv=kfold)

    print(name + ':')
    print(y_pred)

    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('F1 Score:', f1_score(y_test, y_pred, average='macro'))
    print('Precision:', precision_score(y_test, y_pred, average='macro'))
    print('Recall:', recall_score(y_test, y_pred, average='macro'))
    print(
        f'Cross Validation Accuracy: mean={round(results.mean(), 5)}, std={round(results.std(), 5)}\n')

    return y_pred


def pre_process_data(df) -> tuple[DataSet, DataSet, list[str]]:
    data_processor = DataProcessor(df)
    unique_teams = data_processor.get_unique_teams()
    train, test = data_processor.split_data(train_test_ratio=0.95)

    return train, test, unique_teams


def feature_engineering(train: DataSet, test: DataSet, unique_teams, feature_params):
    X_train = XTrainConstructor(
        train.X, unique_teams, **feature_params).construct_table()
    X_train = DataEncoder(X_train).run()
    y_train = TargetMapper(train.y).run()

    X_test = XTestConstructor(
        test.X, train.X, unique_teams, **feature_params).construct_table()
    X_test = DataEncoder(X_test).run()
    y_test = TargetMapper(test.y).run()

    X_train, X_test = DataAligner(X_train, X_test).run()

    return X_train, y_train, X_test, y_test


def run(df, feature_params=get_feature_params()):
    train, test, unique_teams = pre_process_data(df)
    X_train, y_train, X_test, y_test = feature_engineering(
        train, test, unique_teams, feature_params)

    lr = LogisticRegression(max_iter=1000)
    lr = train_model(lr, 'Logistic Regression',
                     X_train, y_train, X_test, y_test)

    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    rf = train_model(rf, 'Random Forest', X_train, y_train, X_test, y_test)

    xgboost = xgb.XGBClassifier(objective='multi:softmax', num_class=3)
    xgboost = train_model(xgboost, 'XGBoost', X_train, y_train, X_test, y_test)

    catboost = CatBoostClassifier(
        iterations=1000, depth=5, learning_rate=0.1, loss_function='MultiClass')
    catboost.set_params(logging_level='Silent')
    catboost = train_model(catboost, 'CatBoost', X_train,
                           y_train, X_test, y_test)

    svm = SVC(kernel='linear')
    svm = train_model(svm, 'SVM', X_train, y_train, X_test, y_test)

    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
    mlp = train_model(mlp, 'MLP Classifier', X_train, y_train, X_test, y_test)

    adb = AdaBoostClassifier(n_estimators=100)
    adb = train_model(adb, 'AdaBoost', X_train, y_train, X_test, y_test)
