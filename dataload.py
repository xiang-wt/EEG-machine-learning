import scipy.io
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
import random
import re
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split

root_path_TS = 'TD/'
root_path_HC = 'HC/'
n_TS = 78 #training and test set
n_HC=77

#significant features screened out after the first statistical analysis
def abstract_feature(folder_path):
    data_list = np.zeros((19,15))
    for root, dirs, files in os.walk(folder_path):         
        for filename in files:
            if filename.startswith('time_features'):
                file_path = os.path.join(root, filename)
                mat_data = scipy.io.loadmat(file_path)
                features = mat_data['time_features'][:, [4,5,6,13]]       
                df = pd.DataFrame(features,columns=['MMAX','skewness','kurtosis','complexity'])
                data_list[:,0:4]=df            
            elif filename.startswith('freq_features_delta'):
                file_path = os.path.join(root, filename)
                mat_data = scipy.io.loadmat(file_path)
                features = mat_data['freq_features'][:, [2, 4]]     
                df = pd.DataFrame(features,columns=['deltaMNF','deltaPSE'])
                data_list[:,4:6]=df
            elif filename.startswith('freq_features_theta'):
                file_path = os.path.join(root, filename)
                mat_data = scipy.io.loadmat(file_path)
                features = mat_data['freq_features'][:, [2, 4]]        
                df = pd.DataFrame(features,columns=['thetaMNF','thetaPSE'])                
                data_list[:,6:8]=df
            elif filename.startswith('freq_features_alpha'):
                file_path = os.path.join(root, filename)
                mat_data = scipy.io.loadmat(file_path)
                features = mat_data['freq_features'][:, [2, 4]]        
                df = pd.DataFrame(features,columns=['alphaMNF','alphaPSE'])
                data_list[:,8:10]=df
            elif filename.startswith('freq_features_beta'):
                file_path = os.path.join(root, filename)
                mat_data = scipy.io.loadmat(file_path)
                features = mat_data['freq_features'][:, [2, 4]]     
                df = pd.DataFrame(features,columns=['betaMNF','betaPSE'])
                data_list[:,10:12]=df
            elif filename.startswith('freq_features_gamma'):
                file_path = os.path.join(root, filename)
                mat_data = scipy.io.loadmat(file_path)
                features = mat_data['freq_features'][:, [2, 4]]      
                df = pd.DataFrame(features,columns=['gammaMNF','gammaPSE'])
                data_list[:,12:14]=df           
            else:
                file_path = os.path.join(root, filename)
                mat_data = scipy.io.loadmat(file_path)
                features = mat_data['tf_features'][:, 4]        
                df = pd.DataFrame(features,columns=['time-frequency entropy'])
                data_list[:,14:15]=df
    data_list=data_list.flatten()
    return data_list     

#match to age and label(0:HC,1:TD)
def get_label_TD():
    TD_dict={}
    TD = np.array(pd.read_excel('XX', sheet_name='TD')[['id_TD','age_TD','TD']])
    for i,j,k in TD:      
        key=int(i)
        value=[j,int(k)]
        TD_dict[key]=value
    return TD_dict

def get_label_HC():
    HC_dict={}
    HC = np.array(pd.read_excel('XX', sheet_name='HC')[['id_HC','age_HC','HC']])    
    for i,j,k in HC:      
        key=int(i)
        value=[j,int(k)]
        HC_dict[key]=value
    return HC_dict

def data_load_personTD(name):
    TD_dict=get_label_TD()            
    folder_path =root_path_TD + name
    data_feature=abstract_feature(folder_path)    
    name_i=int(name[2:])
    for i in TD_dict:
        if i==int(name_i):            
            age,label=TD_dict[i]
    out=np.concatenate((data_feature,[age,label]))   
    return out
           
def data_load_personHC(name):
    HC_dict=get_label_HC()            
    folder_path =root_path_HC + name
    data_feature=abstract_feature(folder_path)    
    name_i=int(name[2:])
    for i in HC_dict:
        if i==int(name_i):            
            age,label=HC_dict[i]
    out=np.concatenate((data_feature,[age,label]))
    return out        

#obtain training set and test set
def dataload():
    #after the second screening by RFECV, remain:True,reject:False
    change= np.array(pd.read_excel('XX',sheet_name='Sheet2')[['c']])
    all_data_TD=[]
    path_name_TD=os.listdir(root_path_TD)
    all_data_HC=[]
    path_name_HC=os.listdir(root_path_HC)
    
    for name in path_name_TD:        
        data=data_load_personTD(name)
        da=[]
        for i in range(len(data)):
            if change[i]==[True]:
                da.append(data[i])        
        da=np.array(da)
        da=da.reshape(1,31) #30 features + label
        all_data_TD=pd.concat([pd.DataFrame(all_data_TD), pd.DataFrame(da)], axis=0)
    for name in path_name_HC:            
        data=data_load_personHC(name)
        da=[]
        for i in range(len(data)):
            if change[i]==[True]:
                da.append(data[i])        
        da=np.array(da)
        da=da.reshape(1,31)        
        all_data_HC=pd.concat([pd.DataFrame(all_data_HC), pd.DataFrame(da)], axis=0)

    all_data=pd.concat([all_data_TD,all_data_HC])    
    X=all_data.loc[:,0:29]
    X=pd.concat([X], ignore_index=True)    
    Y=all_data.loc[:,30]
    Y=pd.concat([Y], ignore_index=True)
    #standardization
    ss = StandardScaler()
    X_normalize = ss.fit_transform(X)
    X_normalize=pd.DataFrame(X_normalize)
    
    #ensure the number of TD and HC in each fold is consistent
    X_TD=X_normalize.loc[:77,:]
    X_TD=np.array(X_TD)
    Y_TD=np.array(Y.loc[:77])   
    X_HC=X_normalize.loc[78:,:]
    X_HC=np.array(X_HC)
    Y_HC=np.array(Y.loc[78:])
    
    skf_TD = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    skf_HC = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
    
    TD_train_index=[]
    TD_test_index=[]
    HC_train_index=[]
    HC_test_index=[]
    
    for i,j in skf_TD.split(X_TD,Y_TD): 
        TD_train_index.append(i)
        TD_test_index.append(j)
    for m,n in skf_HC.split(X_HC,Y_HC):
        HC_train_index.append(m)
        HC_test_index.append(n)

    return X_TD,Y_TD,X_HC,Y_HC,TD_train_index,TD_test_index,HC_train_index,HC_test_index,ss
