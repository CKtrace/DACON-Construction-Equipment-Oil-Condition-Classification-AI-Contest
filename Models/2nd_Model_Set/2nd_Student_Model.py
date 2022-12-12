# PUBLIC SCORE : 0.59121

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import f1_score

with open('teacher_train.pkl','rb') as f:
    teacher_train = pickle.load(f)

with open('teacher_test.pkl','rb') as f:
    teacher_test = pickle.load(f)


def student_data(X, y):
    test_stage_features = ['COMPONENT_ARBITRARY', 'ANONYMOUS_1', 'YEAR' , 'ANONYMOUS_2', 'AG', 'CO', 'CR', 'CU', 'FE', 'H2O', 'MN', 'MO', 'NI', 'PQINDEX', 'TI', 'V', 'V40', 'ZN']
    X = X[test_stage_features]
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, y, test_size=0.1, random_state=2, stratify=y)
    
    return X_train_s, X_test_s, y_train_s, y_test_s


df = pd.read_csv("just_basic_train.csv")
df1 = pd.read_csv("just_basic_test.csv")

X = df.drop('Y_LABEL', axis=1)
y = df['Y_LABEL']

X_train_s, X_test_s, y_train_s, y_test_s = student_data(X, y)

y_train_s = teacher_train
y_train_s = pd.DataFrame(y_train_s)
y_train_s.drop(1, axis=1, inplace=True)

model = LGBMRegressor(random_state=2)
model.fit(X_train_s, y_train_s)

wow = model.predict(X_test_s)
print(wow)
answer = []

for i in wow:
    if i >=0.802:
        answer.append(0)
    else:
        answer.append(1)
        
# print(answer)

print(f1_score(y_test_s, answer, average='macro'))


#--------------------------------------------------------------------#

subm = model.predict(df1)

answer_sub = []

for i in subm:
    if i >=0.802:
        answer_sub.append(0)
    else:
        answer_sub.append(1)

submission=pd.read_csv('sample_submission.csv')  
submission['Y_LABEL']=answer_sub

# submission.to_csv('Sub2.csv', index=False)