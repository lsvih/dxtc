# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from tqdm import tqdm
# from dataset import load_data
#
# X_train, X_val, y_train, y_val, test = load_data()
#
# clf = RandomForestClassifier(n_estimators=200, criterion='gini', min_samples_split=2, max_features='auto',
#                              bootstrap=True, random_state=2018, verbose=2, n_jobs=-1)
# clf.fit(X_train, y_train)
# result = clf.predict(test)
# data = pd.read_csv('./data/submit_sample.csv')
# data['predict'] = result
# data.to_csv('./temp/submission.csv', index=False)
# 0.68
#
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm

from dataset import load_data

X_train, X_val, y_train, y_val, test = load_data()
# xgboost 要求 label 是 range(0, num_class)，故替换
label_set = {90063345: 201245, 89950166: 93252, 89950167: 51440, 99999828: 37146,
             89016252: 36379, 90109916: 26685, 89950168: 23316,
             99999827: 22753, 99999826: 20393, 90155946: 15477, 99999830: 14840,
             99999825: 14323, 89016253: 10019, 89016259: 9095}
label_set = {k: v for v, k in enumerate(label_set.keys())}

for service in label_set.keys():
    y_train[y_train == service] = label_set[service]
    y_val[y_val == service] = label_set[service]

param = {'max_depth': 6, 'silent': 0, 'objective': 'multi:softprob',
         'subsample': 0.9, 'colsample_bytree': 0.9,
         'learning_rate': 0.1, 'nthread': 24, 'min_child_weight': 2,
         'lambda': 1, 'alpha': 0, 'gamma': 0, 'num_class': 15}

train_data = xgb.DMatrix(data=X_train, label=y_train.values.ravel())
val_data = xgb.DMatrix(data=X_val, label=y_val)
print('fitting...')
# model = xgb.train(param, train_data, 800, evals=[(train_data, 'train'), (val_data, 'eval')], early_stopping_rounds=50)
# model = xgb.train(param, train_data, 1000)
# model.save_model('./temp/xgb.model')

model = xgb.Booster({'nthread': 24})
model.load_model("./temp/xgb.model")

test_data = xgb.DMatrix(data=test.values)
print('predict...')
rs = model.predict(test_data)
result = np.argmax(rs, axis=1)
inverse_label = {v: k for v, k in enumerate(label_set.keys())}

for i, v in enumerate(result):
    result[i] = inverse_label[v]

data = pd.read_csv('./data/submit_sample.csv')
same = pd.read_csv('./temp/known.csv')
data['predict'] = result

# Merge duplicate data
for i, user_id in enumerate(tqdm(list(same['user_id'].items()))):
    data.loc[data.user_id == user_id, 'predict'] = same.loc[same.user_id == user_id, 'predict']
data['predict'] = data['predict'].astype(int)
data.to_csv('./temp/submission.csv', index=False)
