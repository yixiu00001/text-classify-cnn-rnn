
# coding: utf-8

# In[1]:




# In[2]:

from time import time


# In[7]:

def doSVCTrainL1(featuresList, labelList, punish):
    from sklearn.svm import LinearSVC
    t0 = time()
    print ("Begin SVC Train!")
    
    svc = LinearSVC(penalty = "l1", C=punish, dual=False)
                #(penalty, loss='squared_hinge', dual=True, tol=0.0001, C=punish, multi_class='ovr' , 
                    #fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=5000)
    svc_model = svc.fit(featuresList, labelList) 
    tt = time() - t0
    print ("SVC Classifier trained in {} seconds".format(round(tt,3)))
    
    return svc_model


# In[8]:

from time import time
def doSVCTrainL2(featuresList, labelList, punish):
    from sklearn.svm import LinearSVC
    t0 = time()
    print ("Begin SVC Train!")
    
    svc = LinearSVC(penalty = "l2", C=punish)
                #(penalty, loss='squared_hinge', dual=True, tol=0.0001, C=punish, multi_class='ovr' , 
                    #fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=5000)
    svc_model = svc.fit(featuresList, labelList) 
    tt = time() - t0
    print ("SVC Classifier trained in {} seconds".format(round(tt,3)))
    
    return svc_model

from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from time import time
import numpy as np
from sklearn.metrics import classification_report
def  rfModelBase(max_depth, criterion, n_estimators, diseasetype_vecs,y_trainAll,diseasetype_vecsTest,y_testAll):
    rf = RandomForestClassifier(max_depth = max_depth,criterion= criterion, n_estimators=n_estimators )
    rf.fit(diseasetype_vecs,y_trainAll )
    print(rf)
    print(classification_report(np.array(y_testAll,dtype="f"),np.array(rf.predict(diseasetype_vecsTest), dtype='f')))
def rfModelTrain(diseasetype_vecs,y_trainAll,diseasetype_vecsTest,y_testAll ):
    for max_depth in [40]:
        for criterion in ['gini', 'entropy']:
            for n_estimators in [10, 20]:
                  rfModelBase(max_depth, criterion, n_estimators,diseasetype_vecs,y_trainAll,diseasetype_vecsTest,y_testAll)
# In[ ]:

# In[ ]:


from sklearn import datasets
from sklearn import svm, grid_search, datasets
from sklearn.metrics import classification_report
from sklearn.svm import SVC

from sklearn.linear_model import Ridge                   #L2正则化
from sklearn.linear_model import Lasso  


# In[39]:

import numpy as np

s = list(np.arange(0.1,3.1,0.1))
print(s)
t = list([10,100,300,500,800,1000])
print(t)
ss = s+ t
print(ss)


# In[80]:

def svcL2(punish, max_iter,multi_class, featuresList, labelList):
    from sklearn.svm import LinearSVC
    t0 = time()
    print ("-------------Begin SVC Train!------------")
    
    svc = LinearSVC(penalty = "l2", C=punish, max_iter = max_iter, multi_class = multi_class)
                #(penalty, loss='squared_hinge', dual=True, tol=0.0001, C=punish, multi_class='ovr' , 
                    #fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=5000)
    svc_model = svc.fit(featuresList, labelList) 
    print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
    tt = time() - t0
    print ("SVC Classifier trained in {} seconds".format(round(tt,3)))
    return svc_model

from sklearn.svm import LinearSVC
from sklearn.externals import joblib
def LinearSVCSearchL2My(vecList, labelList, vecListTest, labelListTest):
    #param_grid={"C": [0.001, 0.01, 0.1, 1.0, 5.0,10.0], "penalty": "l2", "multi_class":["ovr","crammer_singer"],"max_iter":[500,1000,2000]}
    #L2 gridsearch
    #param_grid=[{"C": [0.001, 0.01, 0.1, 1.0, 5.0,10.0,100,500,100], "penalty": ["l2"], "multi_class":["ovr","crammer_singer"],"max_iter":[500,1000,2000]}]
    #L1 gridsearch
    param_grid=[{"C": ss, "penalty": ["l2"],"multi_class":["ovr","crammer_singer"],"max_iter":[500,1000,2000]}]
    best = 0
    bestModel = "svmModels/svm_l1_train_model.m"
    for c in ss:
        #for max_iter in [1000,500,2000]:
        for max_iter in [1000]:
            #for multi_class in ["ovr","crammer_singer"]:
            for multi_class in ["ovr"]:
            
                svc_model = svcL2(c, max_iter,multi_class, vecList, labelList)
                y_true, y_pred = labelListTest, svc_model.predict(vecListTest)
                #print(y_true[0:10], y_pred[0:10])
                res = classification_report(y_true,y_pred)
                resArr = (res.strip().split("\n"))
                score = float(resArr[len(resArr)-1].strip().split(" ")[21])
                print(svc_model)
                print("this score:", score)
                print(res)
                if(score > best):
                    best = score
                    print("===========================best fscore:=====================",best)
                    print(svc_model)
                    print(res)
                    joblib.dump(svc_model, bestModel+"_"+str(c)+"_"+str(max_iter)+"_"+str(multi_class))
                

    return svc_model
#clf1 = LinearSVCSearchL2My(vecList, labelList, vecListTest, labelListTest)

def svcL1(punish, max_iter,multi_class, featuresList, labelList):
    from sklearn.svm import LinearSVC
    t0 = time()
    print ("--------------Begin SVC Train!----------------")
    
    svc = LinearSVC(penalty = "l1", C=punish, dual=False, max_iter = max_iter, multi_class = multi_class)
                #(penalty, loss='squared_hinge', dual=True, tol=0.0001, C=punish, multi_class='ovr' , 
                    #fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=5000)
    svc_model = svc.fit(featuresList, labelList) 
    tt = time() - t0
    print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
    print ("====SVC Classifier trained in {} seconds".format(round(tt,3)))
    return svc_model


from sklearn.svm import LinearSVC
from sklearn.externals import joblib
def LinearSVCSearchL1My(vecList, labelList, vecListTest, labelListTest):
    #param_grid={"C": [0.001, 0.01, 0.1, 1.0, 5.0,10.0], "penalty": "l2", "multi_class":["ovr","crammer_singer"],"max_iter":[500,1000,2000]}
    #L2 gridsearch
    #param_grid=[{"C": [0.001, 0.01, 0.1, 1.0, 5.0,10.0,100,500,100], "penalty": ["l2"], "multi_class":["ovr","crammer_singer"],"max_iter":[500,1000,2000]}]
    #L1 gridsearch
    param_grid=[{"C": ss, "penalty": ["l1"],"multi_class":["ovr","crammer_singer"],"max_iter":[500,1000,2000]}]
    best = 0
    bestModel = "svmModels/svm_l1_train_model.m"
    for c in ss:
        #for max_iter in [1000,500,2000]:
        for max_iter in [1000]:
            #for multi_class in ["ovr","crammer_singer"]:
            for multi_class in ["ovr"]:     
 
                svc_model = svcL1(c, max_iter,multi_class, vecList, labelList)
                y_true, y_pred = labelListTest, svc_model.predict(vecListTest)
                #print(y_true[0:10], y_pred[0:10])
                res = classification_report(y_true,y_pred)
                resArr = (res.strip().split("\n"))
                score = float(resArr[len(resArr)-1].strip().split(" ")[21])
                print(svc_model)
                print("this score:", score)
                print(res)
                if(score > best):
                    best = score
                    print("===========================best fscore================:",best)
                    print(svc_model)
                    print(res)
                          
                    joblib.dump(svc_model, bestModel+"_"+str(c)+"_"+str(max_iter)+"_"+str(multi_class))
                

    return svc_model
#svc_model = LinearSVCSearchL1My(vecList, labelList, vecListTest, labelListTest)

