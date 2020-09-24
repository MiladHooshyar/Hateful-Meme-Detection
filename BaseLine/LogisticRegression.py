from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix

def logistic_regression(X, y, X_val, y_val):
    clf = LogisticRegression(random_state=0).fit(X, y)
    prob = clf.predict_proba(X)
    prob_val = clf.predict_proba(X_val)

    confusion = confusion_matrix(
        y_val, prob_val.argmax(axis=1))

    return roc_auc_score(y, prob[:, 1]),\
           roc_auc_score(y_val, prob_val[:, 1]),\
           confusion