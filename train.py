import warnings
from time import clock

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')
from feature import load_data

MODE = 'TRAIN'  # TEST | TRAIN | LOAD
assert MODE in ['TEST', 'TRAIN', 'LOAD']
print('This is %s mode' % MODE)

train, val, test = load_data()

# 我来解释新测试集数据。 新测试集数据来自于复赛的准备的数据。 初赛跟复赛数据应该刚好跨年了， 根据我的统计89016252，89016259 ，89016253 ，99104722  四类套餐已经没有了，其他套餐的分布情况跟元数据集基本一致。
# 这个是导致目前分数较低的一个原因， 原训练集中的四类套餐大家可以当做噪声数据进行处理。
# 细节的数据处理没有及时公布，大家见谅。

# Split data into 2 pieces by service_type:
service_1_label = [90063345, 90109916, 90155946]
# service_4_label = [89950166, 89950167, 99999828, 89016252, 89950168, 99999827, 99999826, 99999830, 99999825, 89016253,
#                    89016259]
service_4_label = [89950166, 89950167, 99999828, 89950168, 99999827, 99999826, 99999830, 99999825]

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


def f1_score_vali(preds, data_vali):
    labels = data_vali.get_label()
    preds = np.argmax(preds.reshape(15, -1), axis=0)
    score_vali = f1_score(y_true=labels, y_pred=preds, average='weighted')
    return 'f1_score', score_vali, True


model1, model4 = None, None
if MODE == 'TEST':
    # param1 = {'max_depth': 6, 'silent': 1, 'objective': 'multi:softprob', 'num_class': 3}
    param4 = {'learning_rate': 0.05, 'n_estimators': 400, 'max_depth': 6, 'min_child_weight': 1, 'seed': 0,
              'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1, 'n_jobs': -1}
    model4 = xgb.XGBClassifier(**param4)
    cv_params = {'max_depth': [4, 6, 8, 10]}
    optimized_GBM = GridSearchCV(estimator=model4, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)
    optimized_GBM.fit(train_1_X.values, train_1_y.values.ravel())
    print('Best value: {0}'.format(optimized_GBM.best_params_))
    print('Best score: {0}'.format(optimized_GBM.best_score_))
    exit(0)
    # model1 = xgb.train(param1, train_1, 200, evals=[(train_1, 'train'), (val_1, 'eval')],
    #                    early_stopping_rounds=50)
    # model4 = xgb.train(param4, train_4, 200, evals=[(train_4, 'train'), (val_4, 'eval')],
    #                    early_stopping_rounds=50)
elif MODE == 'TRAIN':
    start = clock()
    # cv_pred = None
    # skf = StratifiedKFold(n_splits=10, random_state=2018, shuffle=True)
    # for index, (train_index, test_index) in enumerate(skf.split(X, y)):
    #     print(index)
    #     X_train, X_valid, y_train, y_valid = X[train_index], X[test_index], y[train_index], y[test_index]
    #     train_data = lgb.Dataset(X_train, label=y_train)
    #     validation_data = lgb.Dataset(X_valid, label=y_valid)
    #     clf = lgb.train(params, train_data, num_boost_round=100000, valid_sets=[validation_data],
    #                     early_stopping_rounds=50, feval=f1_score_vali, verbose_eval=1)
    #     xx_pred = clf.predict(X_valid, num_iteration=clf.best_iteration)
    #     xx_pred = [np.argmax(x) for x in xx_pred]
    #     xx_score.append(f1_score(y_valid, xx_pred, average='weighted'))
    #     y_test = clf.predict(X_test, num_iteration=clf.best_iteration)
    #     y_test = [np.argmax(x) for x in y_test]
    #     if index == 0:
    #         cv_pred = np.array(y_test).reshape(-1, 1)
    #     else:
    #         cv_pred = np.hstack((cv_pred, np.array(y_test).reshape(-1, 1)))
    # # vote
    # submit = []
    # for line in cv_pred:
    #     submit.append(np.argmax(np.bincount(line)))
    param1 = {'max_depth': 6, 'silent': 1, 'objective': 'multi:softprob', 'num_class': 3}
    param4 = {'learning_rate': 0.05, 'n_estimators': 400, 'max_depth': 6, 'min_child_weight': 1, 'seed': 0,
              'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1, 'n_jobs': -1,
              'num_class': 8, 'silent': 1, 'objective': 'multi:softprob'}
    print('Fitting model 1...')
    model1 = xgb.train(param1, train_1, 1500)
    model1.save_model('./temp/xgb1.model')
    print('Fitting model 4...')
    model4 = xgb.train(param4, train_4, 1500)
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

result_1['current_service'], result_4['current_service'] = rs_1, rs_4

data = pd.read_csv('./data/submit_sample.csv')
same = pd.read_csv('./temp/known.csv')

rs = pd.concat([result_1, result_4])

data = pd.merge(data, rs, how='left', on='user_id')


def y2x(df):
    if pd.notna(df.current_service_y):
        df.current_service_x = df.current_service_y
    return df


data = data.apply(y2x, axis=1).drop('current_service_y', axis=1).rename(columns={'current_service_x': 'current_service'})

data = pd.merge(data, same, how='left', on='user_id')
data = data.apply(y2x, axis=1).drop('current_service_y', axis=1).rename(columns={'current_service_x': 'current_service'})

data['current_service'] = data['current_service'].astype(int)
data.to_csv('./temp/submission.csv', index=False)


