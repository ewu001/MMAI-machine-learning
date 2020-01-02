import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score



def logistic_regression():
    lr = LogisticRegression()
    return lr

def naive_bayes():
    naive_bayes_clf = GaussianNB()
    return naive_bayes_clf

def random_forest():
    random_forest_clf = RandomForestClassifier()
    return random_forest_clf

def decision_tree():
    decision_tree_clf = DecisionTreeClassifier()
    return decision_tree_clf

def gradient_boosting():
    gradient_boost_clf = GradientBoostingClassifier()
    return gradient_boost_clf

def train_tuning(model, data_X, data_y, hparam={}, cvmetric=""):
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, random_state=42, test_size=0.2)
    print("X_train size: ", X_train.shape)
    if hparam != {}:
        model = GridSearchCV(model, hparam, cv=5, return_train_score=True, scoring=cvmetric)
        
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('Accuracy : ', accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    return model

def train_kfold_cv(model, data_X, data_y, kfold, metric):
    num_folds = kfold
    score = metric
    kfold = KFold(n_splits=num_folds, shuffle=False, random_state=42)
    CV_score = cross_val_score(model, data_X, data_y, cv=kfold, scoring=score)

    print("CV with K-Fold " + metric + " score: %0.3f (+/- %0.2f)" % (CV_score.mean(), CV_score.std() * 2))
    return model

def roc_auc_plot(model, data_X, data_y):
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, random_state=42, test_size=0.2)
    probs = model.predict_proba(X_test)
    preds = probs[:,1]
    fpr, tpr, threshold = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)

    # plot the curve
    plt.figure()
    lw = 2

    plt.plot(fpr, tpr, color='red', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

def skmodel_evaluate(model, eval_X, eval_y):
    y_pred = model.predict(eval_X)
    print('Accuracy : ', accuracy_score(eval_y, y_pred))
    print(confusion_matrix(eval_y, y_pred))