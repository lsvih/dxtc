import warnings

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

from utils import *

warnings.filterwarnings('ignore')

fold = 5
param1 = {'max_depth': 6, 'objective': 'multiclass', 'num_class': 3, "lambda_l1": 0.1, "lambda_l2": 0.2, 'n_jobs': -1}
param4 = {'learning_rate': 0.05, 'max_depth': 6, 'min_child_weight': 1, 'seed': 0,
          'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0, 'reg_lambda': 1, 'n_jobs': -1,
          'num_class': 8, 'objective': 'multiclass', "lambda_l1": 0.1, "lambda_l2": 0.2}
model_path = './temp/lgb/'
if not os.path.exists(model_path):
    os.mkdir(model_path)


def f1_score_vali(preds, data_vali):
    labels = data_vali.get_label()
    if 199929 >= preds.shape[0] > 199900:
        preds = np.argmax(preds.reshape(3, -1), axis=0)
    else:
        preds = np.argmax(preds.reshape(8, -1), axis=0)
    score_vali = f1_score(y_true=labels, y_pred=preds, average='weighted')
    return 'f1_score', score_vali, True


def train_model(X, y, test, params, model_num) -> list:
    print('This is model %d' % model_num)
    X_test = test
    xx_score = []
    cv_pred = []
    skf = StratifiedKFold(n_splits=fold, random_state=2018, shuffle=True)
    for index, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(index, '/', fold)
        X_train, X_valid, y_train, y_valid = \
            X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
        train_data = lgb.Dataset(data=X_train, label=y_train)
        validation_data = lgb.Dataset(data=X_valid, label=y_valid)
        clf = lgb.train(params, train_data, num_boost_round=100000, valid_sets=[validation_data],
                        early_stopping_rounds=50, feval=f1_score_vali, verbose_eval=1)
        clf.save_model(model_path + str(model_num) + '_' + str(index))
        xx_pred = clf.predict(X_valid, num_iteration=clf.best_iteration)
        xx_pred = np.argmax(xx_pred, axis=1)
        xx_score.append(f1_score(y_valid, xx_pred, average='weighted'))
        y_test = clf.predict(X_test, num_iteration=clf.best_iteration)
        y_test = np.argmax(y_test, axis=1)
        if index == 0:
            cv_pred = np.array(y_test).reshape(-1, 1)
        else:
            cv_pred = np.hstack((cv_pred, np.array(y_test).reshape(-1, 1)))
    print(xx_score, np.mean(xx_score))
    submit = []
    for line in cv_pred:
        submit.append(np.argmax(np.bincount(line)))
    return submit


train_1_X, train_1_y, train_4_X, train_4_y, test_1, test_4, service_1_label, service_4_label = data()
rs_1 = train_model(train_1_X, train_1_y, test_1, param1, 1)
rs_4 = train_model(train_4_X, train_4_y, test_4, param4, 4)
for i, v in enumerate(rs_1):
    rs_1[i] = service_1_label[v]
for i, v in enumerate(rs_4):
    rs_4[i] = service_4_label[v]
result_1, result_4 = test_1.reset_index()[['user_id']], test_4.reset_index()[['user_id']]
result_1['current_service'], result_4['current_service'] = rs_1, rs_4
data = pd.read_csv('./data/submit_sample.csv')
# same = pd.read_csv('./temp/known.csv')

rs = pd.concat([result_1, result_4])

data = pd.merge(data, rs, how='left', on='user_id')

data = data.apply(y2x, axis=1).drop('current_service_y', axis=1).rename(
    columns={'current_service_x': 'current_service'})

# data = pd.merge(data, same, how='left', on='user_id')
# data = data.apply(y2x, axis=1).drop('current_service_y', axis=1).rename(
#     columns={'current_service_x': 'current_service'})

data['current_service'] = data['current_service'].astype(int)
data.to_csv('./temp/submission.csv', index=False)
