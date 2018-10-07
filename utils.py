import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def middle(file):
    def wrapper(func):
        def _wrapper(*args, **kargs):
            if os.path.exists(file):
                return pickle.load(open(file, 'rb'))
            else:
                rs = func(*args, **kargs)
                pickle.dump(rs, open(file, 'wb'))
                return rs

        return _wrapper

    return wrapper


def f1_score_vali(preds, data_vali):
    labels = data_vali.get_label()
    preds = np.argmax(preds, axis=1)
    score_vali = f1_score(y_true=labels, y_pred=preds, average='weighted')
    return 'f1_score', -score_vali


def y2x(df):
    if pd.notna(df.current_service_y):
        df.current_service_x = df.current_service_y
    return df


@middle('./temp/data.bin')
def data():
    from feature import load_data
    train, val, test = load_data()
    service_1_label = [90063345, 90109916, 90155946]
    service_4_label = [89950166, 89950167, 99999828, 89950168, 99999827, 99999826, 99999830, 99999825]
    label_set_1 = {k: i for i, k in enumerate(service_1_label)}
    label_set_4 = {k: i for i, k in enumerate(service_4_label)}
    train_1, train_4 = train[train.service_type == 1], train[train.service_type == 4]
    test_1, test_4 = test[test.service_type == 1], test[(test.service_type == 4) | (test.service_type == 3)]
    test_1.drop('service_type', axis=1, inplace=True)
    test_4.drop('service_type', axis=1, inplace=True)
    val_1, val_4 = val[val.service_type == 1], val[val.service_type == 4]
    train_1_X = train_1.drop(['current_service', 'service_type'], axis=1)
    train_4_X = train_4.drop(['current_service', 'service_type'], axis=1)
    val_1_X = val_1.drop(['current_service', 'service_type'], axis=1)
    val_4_X = val_4.drop(['current_service', 'service_type'], axis=1)
    train_1_y, train_4_y = train_1[['current_service']], train_4[['current_service']]
    val_1_y, val_4_y = val_1[['current_service']], val_4[['current_service']]
    for service in label_set_1.keys():
        train_1_y[train_1_y == service] = label_set_1[service]
        val_1_y[val_1_y == service] = label_set_1[service]
    for service in label_set_4.keys():
        train_4_y[train_4_y == service] = label_set_4[service]
        val_4_y[val_4_y == service] = label_set_4[service]
    return train_1_X, train_1_y, train_4_X, train_4_y, test_1, test_4, service_1_label, service_4_label
