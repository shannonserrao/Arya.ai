#!/usr/bin/env python3

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import pickle

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression 

from sklearn.model_selection import GridSearchCV, KFold, train_test_split



from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, f1_score, roc_curve, confusion_matrix


# X_train = pd.read_csv('X_train.csv',index_col='Unnamed: 0')
# y_train = pd.read_csv('y_train.csv')


X_val = pd.read_csv('X_val.csv' ,index_col='Unnamed: 0')
y_val = pd.read_csv('y_val.csv')

new_row = pd.DataFrame({'2793':2793, '0':0}, index=[0])
y_val = pd.concat([new_row,y_val.loc[:]]).reset_index(drop=True)
y_val.columns=['Unnamed: 0','y']

y_val = y_val['y']

def metric_analysis(y_val, y_pred):
    accuracy = accuracy_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred)
    print(f"Accuracy : {round(accuracy,4)} || Recall : {round(recall,4)} || f1_Score : {round(f1,4)} || roc_auc_score :{round(roc_auc,4)}")
    
filename  = "finalized_model_aryaAI.sav"

loaded_model = pickle.load(open(filename, 'rb'))
y_pred = loaded_model.predict(X_val)

print(metric_analysis(y_val, y_pred))

cm=confusion_matrix(y_val, y_pred)
cm.astype(int)

df_cm = pd.DataFrame(cm.astype(int), range(2), range(2))
sns.set(font_scale=1.4) # for label size
hm = sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
plt.xlabel('True values')
plt.ylabel('Predictions')


fig = hm.get_figure()
fig.savefig("heat_map.png")

plt.show()
