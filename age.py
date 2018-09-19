import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

from utils import middle


@middle('./temp/filled_age.bin')
def fill_age(train_data, test_data):
    print('Loading predict age...')
    train_data['age'] = train_data['age'].fillna('-1').astype(int)
    test_data['age'] = test_data['age'].fillna('-1').astype(int)
    train_data.loc[(train_data['age'] >= 90) | (train_data['age'] <= 6), 'age'] = -1
    test_data.loc[(test_data['age'] >= 90) | (test_data['age'] <= 6), 'age'] = -1
    train_data['age'] = train_data['age'].replace(-1, np.nan)
    test_data['age'] = test_data['age'].replace(-1, np.nan)
    miss_age = pd.concat([train_data[train_data['age'].isna()], test_data[test_data['age'].isna()]])
    fill_age = pd.concat([train_data[train_data['age'].notna()], test_data[test_data['age'].notna()]])
    fill_age.drop(['contract_time', 'gender', '2_total_fee', '3_total_fee'], axis=1, inplace=True)
    miss_age.drop(['contract_time', 'gender', 'age', '2_total_fee', '3_total_fee'], axis=1, inplace=True)
    miss_age['age'] = predict_age(fill_age, miss_age)
    for row in tqdm(list(miss_age[['age']].iterrows())):
        user_id, age = row[0], row[1]['age']
        if user_id in train_data.index:
            train_data.loc[train_data.index == user_id, 'age'] = age
        else:
            test_data.loc[test_data.index == user_id, 'age'] = age
    return train_data, test_data


@middle('./temp/age.bin')
def predict_age(train, test):
    print('predict missed age...')
    X_train = train.drop('age', axis=1)
    y_train = train['age']
    rf = RandomForestRegressor(random_state=2018, n_estimators=200, max_depth=5, n_jobs=-1, verbose=2)
    rf.fit(X_train, y_train)
    return rf.predict(test)
