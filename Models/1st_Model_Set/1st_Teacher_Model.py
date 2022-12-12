import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pickle
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

import warnings
warnings.filterwarnings('ignore')

def sigmoid_with_temp(prob, temperature) :
    distill = []
    for i in prob:
        distill.append(1/(1+np.exp(-i/temperature)))
    
    return distill


df = pd.read_csv("just_basic_train.csv")

X = df.drop('Y_LABEL', axis=1)
y = df['Y_LABEL']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2, stratify=y)


ms = StandardScaler()
X_train= ms.fit_transform(X_train)
X_test = ms.transform(X_test)


models = [LGBMClassifier(verbosity=-1, device='gpu', gpu_platform_id=0, gpu_device_id=0, boosting_type='gbdt', min_child_samples=20, num_iterations=1000, random_state=2),
          GradientBoostingClassifier(random_state=2),
          CatBoostClassifier(silent=True, random_state=2),
          AdaBoostClassifier(random_state=2),
          RandomForestClassifier(random_state=2),
          ]

scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

    
models_ens = list(zip(['LC', 'GB', 'Cat', 'AD', 'RF', 'xgb'], models))

model_ens = VotingClassifier(estimators = models_ens, voting = 'soft')
model_ens.fit(X_train, y_train)

a = model_ens.predict(X_test)
print(model_ens.score(X_test, y_test))

print(metrics.classification_report(a, y_test))

prob_train = model_ens.predict_proba(X_train)
prob_test = model_ens.predict_proba(X_test)


loss_train = sigmoid_with_temp(prob_train, 0.5)
loss_test = sigmoid_with_temp(prob_test, 0.5)

with open('teacher_train.pkl','wb') as f:
    pickle.dump(loss_train,f)

with open('teacher_test.pkl','wb') as f:
    pickle.dump(loss_test,f)