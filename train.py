from time import clock

import numpy as np
import pandas as pd
import xgboost as xgb

from dataset import load_data

MODE = 'TRAIN'  # TEST | TRAIN | LOAD
assert MODE in ['TEST', 'TRAIN', 'LOAD']
print('This is %s mode' % MODE)

train, val, test = load_data()

# Split data into 2 pieces by service_type:
service_1_label = [90063345, 90109916, 90155946]
service_4_label = [89950166, 89950167, 99999828, 89016252, 89950168, 99999827, 99999826, 99999830, 99999825, 89016253,
                   89016259]
label_set_1 = {k: i for i, k in enumerate(service_1_label)}
label_set_4 = {k: i for i, k in enumerate(service_4_label)}

train_1, train_4 = train[train.service_type == 1], train[train.service_type == 4]
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

train_1 = xgb.DMatrix(data=train_1_X.values, label=train_1_y.values.ravel())
train_4 = xgb.DMatrix(data=train_4_X.values, label=train_4_y.values.ravel())
val_1 = xgb.DMatrix(data=val_1_X.values, label=val_1_y.values.ravel())
val_4 = xgb.DMatrix(data=val_4_X.values, label=val_4_y.values.ravel())

model1, model4 = None, None
if MODE == 'TEST':
    param1 = {'max_depth': 6, 'silent': 1, 'objective': 'multi:softprob', 'num_class': 3}
    param4 = {'max_depth': 6, 'silent': 1, 'objective': 'multi:softprob', 'num_class': 11}
    model1 = xgb.train(param1, train_1, 200, evals=[(train_1, 'train'), (val_1, 'eval')],
                       early_stopping_rounds=50)
    model4 = xgb.train(param4, train_4, 200, evals=[(train_4, 'train'), (val_4, 'eval')],
                       early_stopping_rounds=50)
elif MODE == 'TRAIN':
    start = clock()
    param1 = {'max_depth': 6, 'silent': 1, 'objective': 'multi:softprob', 'num_class': 3}
    param4 = {'max_depth': 6, 'silent': 1, 'objective': 'multi:softprob', 'num_class': 11}
    print('Fitting model 1...')
    model1 = xgb.train(param1, train_1, 1200)
    model1.save_model('./temp/xgb1.model')
    print('Fitting model 4...')
    model4 = xgb.train(param4, train_4, 1200)
    model4.save_model('./temp/xgb4.model')
    finish = clock()
    print("{:10.6} s".format(finish - start))
elif MODE == 'LOAD':
    model1, model4 = xgb.Booster({'nthread': 24}), xgb.Booster({'nthread': 24})
    model1.load_model('./temp/xgb1.model')
    model4.load_model('./temp/xgb4.model')

# Predict
test_1, test_4 = test[test.service_type == 1], test[test.service_type == 4]
test_1.drop('service_type', axis=1, inplace=True)
test_4.drop('service_type', axis=1, inplace=True)
result_1, result_4 = test_1.reset_index()[['user_id']], test_4.reset_index()[['user_id']]
test_1, test_4 = xgb.DMatrix(data=test_1.values), xgb.DMatrix(data=test_4.values)
print('predict...')
rs_1, rs_4 = model1.predict(test_1), model4.predict(test_4)
rs_1, rs_4 = np.argmax(rs_1, axis=1), np.argmax(rs_4, axis=1)

for i, v in enumerate(rs_1):
    rs_1[i] = service_1_label[v]
for i, v in enumerate(rs_4):
    rs_4[i] = service_4_label[v]

result_1['predict'], result_4['predict'] = rs_1, rs_4

data = pd.read_csv('./data/submit_sample.csv')
same = pd.read_csv('./temp/known.csv')

rs = pd.concat([result_1, result_4])

data = pd.merge(data, rs, how='left', on='user_id')


def y2x(df):
    if pd.notna(df.predict_y):
        df.predict_x = df.predict_y
    return df


data = data.apply(y2x, axis=1).drop('predict_y', axis=1).rename(columns={'predict_x': 'predict'})

data = pd.merge(data, same, how='left', on='user_id')
data = data.apply(y2x, axis=1).drop('predict_y', axis=1).rename(columns={'predict_x': 'predict'})

data['predict'] = data['predict'].astype(int)
data.to_csv('./temp/submission.csv', index=False)
