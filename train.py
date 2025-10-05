import numpy as np
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_curve, auc, classification_report, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
import matplotlib.cm as cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from xgboost import XGBClassifier
from scipy import stats
import logging
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.preprocessing import  OneHotEncoder
from sklearn.ensemble import VotingClassifier,StackingClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.svm import SVC 
from sklearn.datasets import make_classification 
from sklearn.model_selection import train_test_split 
import shap
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import time
import shap.plots._bar 

def get_TpTN_FpFn(list1, list2):
    #list1:real label,list2:predicted label
    reallabel_list = list1
    predictlabel_list = list2
    TP_count = 0
    TN_count = 0
    FP_count = 0
    FN_count = 0
    for i in range(len(reallabel_list)):
        if reallabel_list[i] == 1 and predictlabel_list[i] == 1:
            TP_count += 1
        if reallabel_list[i] == 0 and predictlabel_list[i] == 0:
            TN_count += 1
        if reallabel_list[i] == 0 and predictlabel_list[i] == 1:
            FP_count += 1
        if reallabel_list[i] == 1 and predictlabel_list[i] == 0:
            FN_count += 1
    return TP_count, TN_count, FP_count, FN_count

def cal_metrics(list1, list2):
    TP_count, TN_count, FP_count, FN_count = get_TpTN_FpFn(list1, list2)
    F2_score = (5 * TP_count) / (5 * TP_count + FP_count + 4*FN_count)#we think misjudging TD is more serious than misjudging HC.    
    ACC = (TP_count + TN_count) / (TP_count + FN_count + TN_count + FP_count)
    SEN = TP_count / (TP_count + FN_count)
    SPE = TN_count / (TN_count + FP_count)
    PRE = TP_count / (TP_count + FP_count)
    return F2_score, ACC, SEN, SPE, PRE

def tscore(metrics,confidence = 0.95):
	return stats.norm.interval(confidence,
							 loc=metrics.mean(),
							 scale=stats.sem(metrics))

def my_logistic():
    logistic_metrics_test = {'F2_score': [], 'ACC': [], 'SEN': [], 'SPE': [], 'PRE': [], 'AUROC': []}    
    
    for k in range(5):
        X_TD_train, X_TD_test = X_TD[TD_train_index[k]], X_TD[TD_test_index[k]]
        Y_TD_train, Y_TD_test = Y_TD[TD_train_index[k]], Y_TD[TD_test_index[k]]
        X_HC_train, X_HC_test = X_HC[HC_train_index[k]], X_HC[HC_test_index[k]]
        Y_HC_train, Y_HC_test = Y_HC[HC_train_index[k]], Y_HC[HC_test_index[k]]
        
        X_train=np.vstack((X_TD_train,X_HC_train))
        X_test=np.vstack((X_TD_test,X_HC_test))
        Y_train=np.hstack((Y_TD_train,Y_HC_train))
        Y_test=np.hstack((Y_TD_test,Y_HC_test))
  
        pred_test_label_list = []
        real_test_label_list = []

        logistic = LogisticRegression(solver='lbfgs')
        logistic.fit(X_train, Y_train)

        pred_test_label_list.append(logistic.predict(X_test))
        real_test_label_list.append(Y_test)
        #AUROC
        auroc_test = roc_auc_score(Y_test, logistic.predict(X_test))
        #calculate test set indicators
        j = 0
        for key in logistic_metrics_test.keys():
            if key == 'AUROC':
                logistic_metrics_test[key].append(auroc_test)
            else:
                logistic_metrics_test[key].append(cal_metrics(real_test_label_list[0], pred_test_label_list[0])[j])
            j += 1

    return logistic_metrics_test

X_TD,Y_TD,X_HC,Y_HC,TD_train_index,TD_test_index,HC_train_index,HC_test_index,ss = dataload()
logistic_metrics_test = my_logistic()
for key in logistic_metrics_test.keys():
	print(key, ':', round(np.mean(np.array(logistic_metrics_test[key])),3), tscore(np.array(logistic_metrics_test[key])))
        
#SHAP analysis
def my_logistic_SHAP():
    logistic_metrics_test = {'F2_score': [], 'ACC': [], 'SEN': [], 'SPE': [], 'Precision': [], 'AUROC': []}
    all_shap_values = []
    all_expected_values = []
    all_X_test = []
    all_Y_test=[]

    for k in range(5):
        X_TD_train, X_TD_test = X_TD[TD_train_index[k]], X_TD[TD_test_index[k]]
        Y_TD_train, Y_TD_test = Y_TD[TD_train_index[k]], Y_TD[TD_test_index[k]]
        X_HC_train, X_HC_test = X_HC[HC_train_index[k]], X_HC[HC_test_index[k]]
        Y_HC_train, Y_HC_test = Y_HC[HC_train_index[k]], Y_HC[HC_test_index[k]]
        
        X_train=np.vstack((X_TD_train,X_HC_train))
        X_test=np.vstack((X_TD_test,X_HC_test))
        Y_train=np.hstack((Y_TD_train,Y_HC_train))
        Y_test=np.hstack((Y_TD_test,Y_HC_test))

        pred_test_label_list = []
        real_test_label_list = []

        logistic =LogisticRegression(solver='lbfgs') 
        logistic.fit(X_train, Y_train)
        
        pred_test_label_list.append(logistic.predict(X_test))
        real_test_label_list.append(Y_test)
        
        #AUROC
        auroc_test = roc_auc_score(Y_test, logistic.predict(X_test))
        #calculate test set indicators
        j = 0
        for key in logistic_metrics_test.keys():
            if key == 'AUROC':
                logistic_metrics_test[key].append(auroc_test)
            else:
                logistic_metrics_test[key].append(cal_metrics(real_test_label_list[0], pred_test_label_list[0])[j])

            j += 1

        # SHAP result        
        explainer = shap.KernelExplainer(logistic.predict_proba,X_train)
        shap_values = explainer.shap_values(X_test)
        
        all_shap_values.append(shap_values)
        all_expected_values.append(explainer.expected_value)
        all_X_test.append(X_test)
        all_Y_test.append(Y_test)
                
    shap_values = np.concatenate(all_shap_values, axis=1)
    expected_values = np.mean(all_expected_values, axis=0)
    X_test = np.concatenate(all_X_test, axis=0)
    Y_test = np.concatenate(all_Y_test, axis=0)

    return logistic_metrics_test,shap_values,expected_values,X_test,Y_test

logistic_metrics_test,shap_values,expected_values,X_test,Y_test=my_logistic_SHAP()

#Weight
columns=["Fp1_skewness","Fz_skewness","F4_MMAX","F4_complexity","F4_PSE_γ","C3_complexity","C3_MNF_γ","Cz_complexity","C4_complexity","P3_complexity","P3_MNF_β","Pz_complexity","P4_skewness","P4_complexity","O1_MMAX","O1_skewness","O1_PSE_γ","O2_MMAX","O2_skewness","O2_kurtosis","O2_PSE_γ","F7_skewness","F7_kurtosis","F7_MNF_γ","F8_PSE_γ","T3_skewness","T3_complexity","T3_PSE_γ","T5_complexity","age"]
shap.initjs()
all_shap_values=shap.Explanation(shap_values[1], base_values=expected_values[1], data=X_test, feature_names=columns)
a=pd.DataFrame(all_shap_values.values)
b=abs(a).mean(axis=0)
weight=b/sum(b)