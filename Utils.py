from sklearn import metrics

def roc(actual, predicted):
    fpr, tpr, _ = metrics.roc_curve(actual, predicted)
    print("AUC: {0}", metrics.auc(fpr, tpr))