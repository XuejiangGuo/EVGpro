import pandas as pd
import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# load data
data = pd.read_csv('data.csv')
# feature
x = data.iloc[:, 2:]
# label
y = data['Type'].map({'AZS_C1': 0, 'AZS_C2': 1})  

# the function for plot ROC
def plotROC(y_true, y_prob, name):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='{} (AUC = {})'.format(name, round(roc_auc, 3)))

# load RandomForest model
clf = RandomForestClassifier(n_jobs=-1, class_weight="balanced")

# RFECV for select importance features
cv = StratifiedKFold(5)
selector = RFECV(clf, step=1, cv=cv, n_jobs=-1,scoring='roc_auc')
selector = selector.fit(x, y)
select_features = selector.get_feature_names_out()

# plot ROC for each importance features
plt.rcParams['font.sans-serif'] = 'Arial'
plt.figure(figsize=(4.5,4))
for feature in select_features:
    x_s = x[[feature]]
    y_s_pred = cross_val_predict(clf, x_s, y, cv=cv, method= 'predict_proba')[:,1]
    plotROC(y, y_s_pred, feature)

# plot ROC for all importance features
y_pred = cross_val_predict(clf, x[select_features], y,
                           cv=cv, method= 'predict_proba')[:,1]
plotROC(y, y_pred, 'All proteins')

plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.legend(loc="lower right")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('AUC.pdf')