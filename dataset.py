import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from age import fill_age


# Dataset counter:
# 90063345: 201245, 89950166: 93252, 89950167: 51440, 99999828: 37146,
# 89016252: 36379, 99104722: 36289, 90109916: 26685, 89950168: 23316,
# 99999827: 22753, 99999826: 20393, 90155946: 15477, 99999830: 14840,
# 99999825: 14323, 89016253: 10019, 89016259: 9095


def load_data():
    print('Loading dataset...')
    train_data = pd.read_csv('./data/train.csv', index_col='user_id')
    test_data = pd.read_csv('./data/test.csv', index_col='user_id')
    one_hot_feature = ['contract_type', 'complaint_level', 'gender', 'net_service', 'is_mix_service',
                       'many_over_bill', 'is_promise_low_consume']
    range_features = ['online_time', '1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee', 'month_traffic',
                      'local_trafffic_month', 'local_caller_time', 'service1_caller_time', 'service2_caller_time',
                      'age', 'former_complaint_fee', 'contract_time', 'pay_times', 'pay_num', 'last_month_traffic',
                      'former_complaint_num', 'service_type']  # service_type is a one hot feature

    if not os.path.exists('./temp/known.csv'):
        type_3_test = test_data[test_data['service_type'] == 3].reset_index()[['user_id']]
        type_3_test['predict'] = 99104722
        train_test_same = \
            pd.merge(train_data.reset_index(), test_data.reset_index(), how='inner', on=list(test_data.columns)[:-1])[
                ['user_id_y', 'current_service']].rename(
                columns={'user_id_y': 'user_id', 'current_service': 'predict'})
        known_test = type_3_test.append(train_test_same).drop_duplicates('user_id')
        known_test.to_csv('./temp/known.csv', index=False)

    # There is nobody with service type 3.
    # Drop service type == 3
    train_data = train_data[train_data.service_type != 3]
    test_data = test_data[test_data.service_type != 3]

    # Process invalid values.
    train_data = train_data.replace('\\N', np.nan)
    test_data = test_data.replace('\\N', np.nan)

    train_data = train_data[~train_data.iloc[:, :-2].duplicated(keep=False)]
    train_data = train_data[train_data['2_total_fee'].notna()]
    train_data[['2_total_fee', '3_total_fee']] = train_data[['2_total_fee', '3_total_fee']].astype(float)
    test_data[['2_total_fee', '3_total_fee']] = test_data[['2_total_fee', '3_total_fee']].astype(float)
    train_data.loc[train_data.contract_time == -1, 'contract_time'] = np.nan
    test_data.loc[test_data.contract_time == -1, 'contract_time'] = np.nan
    labels = train_data[['current_service']]
    train_data = train_data.drop('current_service', axis=1)

    # Fix invalid gender
    train_data['gender'] = train_data['gender'].fillna('0')
    train_data['gender'] = train_data['gender'].replace('00', '0')
    train_data['gender'] = train_data['gender'].replace('01', '1')
    train_data['gender'] = train_data['gender'].replace('02', '2')
    train_data['gender'] = train_data['gender'].astype(int)

    test_data['gender'] = test_data['gender'].fillna('0')
    test_data['gender'] = test_data['gender'].replace('00', '0')
    test_data['gender'] = test_data['gender'].replace('01', '1')
    test_data['gender'] = test_data['gender'].replace('02', '2')
    test_data['gender'] = test_data['gender'].astype(int)

    # Fill missed age
    train_data, test_data = fill_age(train_data, test_data)

    train_data[['contract_type', 'net_service', 'complaint_level']] \
        = train_data[['contract_type', 'net_service', 'complaint_level']].astype(str)
    test_data[['contract_type', 'net_service', 'complaint_level']] \
        = test_data[['contract_type', 'net_service', 'complaint_level']].astype(str)

    train, test = train_data[range_features].astype('float64'), test_data[range_features].astype('float64')

    enc = OneHotEncoder()
    for feature in one_hot_feature:
        enc.fit(train_data[feature].values.reshape(-1, 1))
        train_a = enc.transform(train_data[feature].values.reshape(-1, 1)).toarray()
        test_a = enc.transform(test_data[feature].values.reshape(-1, 1)).toarray()
        train_b, test_b = pd.DataFrame(train_a, index=train.index), pd.DataFrame(test_a, index=test.index)
        train_b.columns, test_b.columns = [list(map(lambda x: feature + '_' + str(x), range(train_a.shape[1])))] * 2
        train, test = pd.concat([train, train_b], axis=1), pd.concat([test, test_b], axis=1)

    dataset = pd.concat([train, labels], axis=1)
    train, val = train_test_split(dataset, test_size=0.23, random_state=42)
    return train, val, test


if __name__ == '__main__':
    load_data()
