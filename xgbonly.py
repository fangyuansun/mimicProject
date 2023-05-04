# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import load_iris
#from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
#from sklearn.metrics import plot_roc_curve


patient_filePath='D:/physionet.org/files/concepts/mytemp2.csv'

newdata = pd.read_csv(patient_filePath)
newdata["gender"]=newdata["gender"].replace(["M","F"],[1,0])
# from sklearn.preprocessing import OneHotEncoder
# X=patient.loc[:,"gender"].values.reshape(-1,1)
# enc = OneHotEncoder(categories='auto').fit(X)
# result = enc.transform(X).toarray()
# newdata = pd.concat([patient,pd.DataFrame(result)],axis=1)

#for i in range(len(newdata.columns)):
#    print(newdata.columns[i],newdata.iloc[:,i].isnull().sum())

from sklearn.preprocessing import label_binarize


newdata.drop(["stay_id"],axis=1,inplace=True)
newdata.drop(["hadm_id"],axis=1,inplace=True)
newdata.drop(["survive_days"],axis=1,inplace=True)
newdata.drop(["survive_28_days"],axis=1,inplace=True)
newdata.drop(["survive_90_days"],axis=1,inplace=True)
newdata.drop(["survive_360_days"],axis=1,inplace=True)
newdata.drop(["invasiveventtime"],axis=1,inplace=True)
newdata.drop(["noninvasiveventtime"],axis=1,inplace=True)
newdata.drop(["vaso_total"],axis=1,inplace=True)
newdata.drop(["los"],axis=1,inplace=True)
#newdata.drop(["antibiotic"],axis=1,inplace=True)
#去掉相关性比较强的项
#去掉干扰特征
newdata.drop(["smoker"],axis=1,inplace=True)
newdata.drop(["etoh"],axis=1,inplace=True)
#newdata.drop(["dbp"],axis=1,inplace=True)





for i in range(len(newdata.columns)):
    if newdata.columns[i]=="fio2" or newdata.columns[i]=="fio2":
        newdata.loc[:,newdata.columns[i]].fillna(21,inplace=True)
    elif newdata.columns[i]=="peep" or newdata.columns[i]=="peep":
        newdata.loc[:,newdata.columns[i]].fillna(0,inplace=True)
    elif newdata.columns[i]=="tidal" or newdata.columns[i]=="tidal":
        newdata.loc[:,newdata.columns[i]].fillna(0,inplace=True)
    elif newdata.columns[i]=="vaso_total" \
    or newdata.columns[i]=="vaso_24hour" \
    or newdata.columns[i]=="cad"\
    or newdata.columns[i]=="hypertension"\
    or newdata.columns[i]=="diabetes" \
    or newdata.columns[i]=="liver"\
    or newdata.columns[i]=="etoh" \
    or newdata.columns[i]=="renal" \
    or newdata.columns[i]=="smoker"\
    or newdata.columns[i]=="copd"\
    or newdata.columns[i]=="rrt":
        newdata.loc[:,newdata.columns[i]].fillna(0,inplace=True)
        
print("before 25%%",newdata.shape[0])       
for index, row in newdata.iterrows():
    if newdata.loc[index].isna().sum()>(newdata.columns.shape[0]-3)*0.25:
        newdata.drop(index=index,axis=0,inplace=True)
        continue
 
print("after 25%%",newdata.shape[0])
    
for i in range(len(newdata.columns)):
    if newdata.columns[i]=="weight":
        weight_male_median = newdata.loc[newdata["gender"]==1,"weight"].median()
        weight_female_median = newdata.loc[newdata["gender"]==0,"weight"].median()
        newdata.loc[newdata["gender"]==1,"weight"]=newdata.loc[newdata["gender"]==1,"weight"].fillna(weight_male_median)
        newdata.loc[newdata["gender"]==0,"weight"]=newdata.loc[newdata["gender"]==0,"weight"].fillna(weight_female_median)      
    elif newdata.columns[i]=="height":
        height_male_median = newdata.loc[newdata["gender"]==1,"height"].median()
        height_female_median = newdata.loc[newdata["gender"]==0,"height"].median()
        newdata.loc[newdata["gender"]==1,"height"].fillna(height_male_median,inplace=True)
        newdata.loc[newdata["gender"]==0,"height"].fillna(height_female_median,inplace=True)
       
for index, row in newdata.iterrows():
    if newdata.loc[index].loc["fio2"]>100 or\
       newdata.loc[index].loc["glucose"]>9999 or\
       newdata.loc[index].loc["baseexcess"]<-400 or\
       newdata.loc[index].loc["temperature"]<24 or\
       newdata.loc[index].loc["weight"]<20 or\
       newdata.loc[index].loc["heart_rate"]<6 or\
       newdata.loc[index].loc["sbp"]<10 or\
       newdata.loc[index].loc["sbp"]>300 :
           # newdata.loc[index].loc["dbp_max"]>300 or\
        newdata.drop(index=index,axis=0,inplace=True)
    # else:
    #     newdata.loc[index,"bmi"]=newdata.loc[index,"weight"]/(newdata.loc[index,"height"]*newdata.loc[index,"height"])
    
print("after other abnormal",newdata.shape[0])   

for i in range(len(newdata.columns)):
    newdata.loc[:,newdata.columns[i]] = newdata.loc[:,newdata.columns[i]].fillna(newdata.loc[:,newdata.columns[i]].median())
newdata.to_csv('D:/physionet.org/files/concepts/mytemp2_filled.csv') 
data_sap = newdata.loc[:,["sapsii"]]
data_gcs = newdata.loc[:,["gcs"]]
data_sofa = newdata.loc[:,["sofa"]]
data_oasis = newdata.loc[:,["oasis"]]
print("shape",newdata.shape[1])
target = newdata.loc[:,"dieinhosp"].values.reshape(-1,1)
data=newdata.drop(["dieinhosp"],axis=1,inplace=False)
data=data.drop(["oasis"],axis=1,inplace=False)
data=data.drop(["sapsii"],axis=1,inplace=False)
data=data.drop(["sofa"],axis=1,inplace=False)
data=data.rename(columns={"weight": "Weight", "gcs": "GCS"
                     ,"peep": "PEEP", "bun": "BUN"
                     ,"aniongap": "Aniongap", "tidal": "Tidal volume"
                     ,"alp": "ALP", "resp_rate": "Respiratory Rate"
                     ,"sbp": "SBP", "heart_rate": "Heart Rate"
                     ,"lactate": "Lactate", "platelets": "Platelets"
                     ,"fio2": "Fio2", "temperature": "Temperature"
                     ,"pt": "PT", "spo2": "Spo2"
                     ,"vent_24time": "24hTV", "ldh": "LDH"
                     ,"ph": "PH", "age": "Age","mbp":"MBP"
                     ,"glucose":"Glucose","dbp":"DBP","hypertension":"Hypertension"
                     ,"creatinine":"Creatinine","antibiotic":"Antibiotic","ptt":"PTT"
                     ,"albumin":"Albumin","pao2fio2ratio":"P/F"})


Xtrain, Xtest, Ytrain, Ytest = train_test_split(data,target,test_size=0.3)
Xtrain_sap, Xtest_sap, Ytrain_sap, Ytest_sap = train_test_split(data_sap,target,test_size=0.3)
Xtrain_gcs, Xtest_gcs, Ytrain_gcs, Ytest_gcs = train_test_split(data_gcs,target,test_size=0.3)
Xtrain_sofa, Xtest_sofa, Ytrain_sofa, Ytest_sofa = train_test_split(data_sofa,target,test_size=0.3)
Xtrain_oasis, Xtest_oasis, Ytrain_oasis, Ytest_oasis = train_test_split(data_oasis,target,test_size=0.3)
Xtrain=Xtrain.rename(columns={"weight": "Weight", "gcs": "GCS"
                     ,"peep": "PEEP", "bun": "BUN"
                     ,"aniongap": "Aniongap", "tidal": "Tidal volume"
                     ,"alp": "ALP", "resp_rate": "Respiratory Rate"
                     ,"sbp": "SBP", "heart_rate": "Heart Rate"
                     ,"lactate": "Lactate", "platelets": "Platelets"
                     ,"fio2": "Fio2", "temperature": "Temperature"
                     ,"pt": "PT", "spo2": "Spo2"
                     ,"vent_24time": "24hTV", "ldh": "LDH"
                     ,"ph": "PH", "age": "Age","mbp":"MBP"
                     ,"glucose":"Glucose","dbp":"DBP"
                     ,"creatinine":"Creatinine","antibiotic":"Antibiotic","ptt":"PTT"
                     ,"albumin":"Albumin","pao2fio2ratio":"P/F","hypertension":"Hypertension"})
Xtest=Xtest.rename(columns={"weight": "Weight", "gcs": "GCS"
                     ,"peep": "PEEP", "bun": "BUN"
                     ,"aniongap": "Aniongap", "tidal": "Tidal volume"
                     ,"alp": "ALP", "resp_rate": "Respiratory Rate"
                     ,"sbp": "SBP", "heart_rate": "Heart Rate"
                     ,"lactate": "Lactate", "platelets": "Platelets"
                     ,"fio2": "Fio2", "temperature": "Temperature"
                     ,"pt": "PT", "spo2": "Spo2"
                     ,"vent_24time": "24hTV", "ldh": "LDH"
                     ,"ph": "PH", "age": "Age","mbp":"MBP"
                     ,"glucose":"Glucose","dbp":"DBP"
                     ,"creatinine":"Creatinine","antibiotic":"Antibiotic","ptt":"PTT"
                     ,"albumin":"Albumin","pao2fio2ratio":"P/F","hypertension":"Hypertension"})


import xgboost as xgb


from sklearn.metrics import roc_curve, auc

# 参数
params = {
  'booster': 'gbtree',
  'learning_rate': 0.05,  # 步长✔
  'max_depth': 6,  # 树的最大深度✔
  'objective': 'multi:softprob',
  'num_class': 2,
  'min_child_weight': 1.0,  # ✔决定最小叶子节点样本权重和，如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束。
  'gamma': 0,  # ✔指定了节点分裂所需的最小损失函数下降值。这个参数的值越大，算法越保守
   # 输出运行信息
  'nthread': 4,
  'seed': 27,
}
num_rounds = 300
# 构造训练集
dtrain_xgb = xgb.DMatrix(Xtrain,Ytrain)
# xgboost模型训练
model = xgb.train(params,dtrain_xgb,num_rounds)
# 对测试集进行预测
dtest_xgb = xgb.DMatrix(Xtest)
Ypred = model.predict(dtest_xgb)


#sap单因素预测
dtrain = xgb.DMatrix(Xtrain_sap,Ytrain_sap)
model_sap = xgb.train(params,dtrain,num_rounds)
dtest = xgb.DMatrix(Xtest_sap)
Ypred_sap = model_sap.predict(dtest)
#gcs单因素预测
dtrain = xgb.DMatrix(Xtrain_gcs,Ytrain_gcs)
model_gcs = xgb.train(params,dtrain,num_rounds)
dtest = xgb.DMatrix(Xtest_gcs)
Ypred_gcs = model_gcs.predict(dtest)
#sofa单因素预测
dtrain = xgb.DMatrix(Xtrain_sofa,Ytrain_sofa)
model_sofa = xgb.train(params,dtrain,num_rounds)
dtest = xgb.DMatrix(Xtest_sofa)
Ypred_sofa = model_sofa.predict(dtest)
#oasis单因素预测
dtrain = xgb.DMatrix(Xtrain_oasis,Ytrain_oasis)
model_oasis = xgb.train(params,dtrain,num_rounds)
dtest = xgb.DMatrix(Xtest_oasis)
Ypred_oasis = model_oasis.predict(dtest)

fpr = dict()
tpr = dict()
thpf = dict()
thrpf = dict()
roc_auc = dict()
Jpf = dict()
plt.figure(figsize=(6,6),dpi=300)
lw = 1
label=["XGB","SAPSII","GCS","SOFA","OASIS"]

pred=[Ypred,Ypred_sap,Ypred_gcs,Ypred_sofa,Ypred_oasis]
predY=[Ytest,Ytest_sap,Ytest_gcs,Ytest_sofa,Ytest_oasis]
predcolor=["red","darkorange","purple","blue","darkgreen","darkred","brown"]
for j in range(len(pred)):
    for i in range(1):
        fpr[i], tpr[i], thpf[i] = roc_curve(predY[j][:, i], pred[j][:,i+1])
        roc_auc[i] = auc(fpr[i], tpr[i])
    Jpf[j] = tpr[0] - fpr[0]
    thrpf[j] = thpf[0]
    plt.plot(
        fpr[0],
        tpr[0],
        color=predcolor[j],
        lw=lw,
        label="%s (AUC = %0.3f)" %(label[j], roc_auc[0])
    )
    #sns.regplot(x=fpr[0], y=tpr[0], ci=95)
#pd.DataFrame(Ypred[:,1]>thrpf[0][argmax(Jpf[0])]).value_counts()
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel("1-Specificity")
plt.ylabel("Sensitivity")
#plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
plt.savefig('D:/physionet.org/files/pic/XBGvsALL.tif')
plt.show()


import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(data)

plt.figure(figsize=(6,6), dpi=300)
# plt.subplot(131)
shap.summary_plot(shap_values[1], data,show = False,max_display=20)
plt.savefig('D:/physionet.org/files/pic/左边.tif')
plt.figure(figsize=(6,6), dpi=300)
shap.summary_plot(shap_values[1], data,show = False,plot_type="bar",max_display=20)
plt.savefig('D:/physionet.org/files/pic/右边.tif')
from PIL import Image
left = Image.open('D:/physionet.org/files/pic/左边.tif')
right = Image.open('D:/physionet.org/files/pic/右边.tif')
plt.xticks([])
plt.yticks([])
plt.figure(figsize=(6,5), dpi=300)
plt.subplot(121) 
plt.imshow(left)
plt.axis("off")
plt.subplot(122) 
plt.imshow(right)

plt.axis("off")
plt.savefig("D:/physionet.org/files/pic/summary_all.tif")
left.close()
right.close()
#load model
#Xtrain = pd.read_csv('D:/physionet.org/files/Xtrain2.csv').iloc[:,1:]
#Xtest = pd.read_csv('D:/physionet.org/files/Xtest2.csv').iloc[:,1:]
#Ytrain = pd.read_csv('D:/physionet.org/files/Ytrain2.csv').iloc[:,1:].values
#Ytest = pd.read_csv('D:/physionet.org/files/Ytest2.csv').iloc[:,1:].values
#model = xgb.Booster()
#model.load_model('D:/physionet.org/files/model2.json')
#Xtrain.to_csv('D:/physionet.org/files/Xtrain2.csv')
#pd.DataFrame(Ytrain).to_csv('D:/physionet.org/files/Ytrain2.csv')
#Xtest.to_csv('D:/physionet.org/files/Xtest2.csv')
#pd.DataFrame(Ytest).to_csv('D:/physionet.org/files/Ytest2.csv')
#model.save_model('D:/physionet.org/files/model2.json')
from sklearn.svm import SVC
from time import time 
import datetime
from sklearn.preprocessing import StandardScaler 
X_svm=pd.concat([Xtrain,Xtest],axis=0)
X_svm = StandardScaler().fit_transform(X_svm) 

X_svm = pd.DataFrame(X_svm)
Xtrain_svm = X_svm.loc[0:round(X_svm.shape[0]*0.70)-2]
Xtest_svm = X_svm.loc[round(X_svm.shape[0]*0.70)-1:]
# Xtest_svm = StandardScaler().fit_transform(Xtest) 
# Xtest_svm = pd.DataFrame(Xtest_svm)

# #score = []
# #C_range = np.linspace(0.01,20,30) 
# #gamma_range = np.logspace(-10, 1, 50) #返回在对数刻度上均匀间隔的数
svc= SVC(kernel = "rbf", gamma=0.020235896477251554, C=1.3886206896551723,cache_size=5000,probability=True).fit(Xtrain_svm,Ytrain.ravel())
# #clf= SVC(kernel="sigmoid",gamma="auto",degree = 1,cache_size=5000).fit(Xtrain,Ytrain)  
Ypred_svm = svc.predict_proba(Xtest_svm)

from sklearn.neural_network import MLPClassifier as DNN 

dnn = DNN(hidden_layer_sizes=(1000,),random_state=420).fit(Xtrain,Ytrain)
Ypred_dnn=dnn.predict_proba(Xtest)
from sklearn.linear_model import LogisticRegression as LR 
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score,multilabel_confusion_matrix
lrl2 = LR(penalty="l2",solver="liblinear",C=1,max_iter=1000)
lrl2 = lrl2.fit(Xtrain,Ytrain.ravel())
Ypred_lr=lrl2.predict_proba(Xtest)
fpr = dict()
tpr = dict()
th = dict()
thr = dict()
roc_auc = dict()
J = dict()
plt.figure(figsize=(6,6),dpi=300)
lw = 1
label=["XGB","LR","SVM","DNN"]
pred=[Ypred,Ypred_lr,Ypred_svm,Ypred_dnn]
predY=[Ytest,Ytest,Ytest,Ytest]

for j in range(len(label)):
    for i in range(1):
        fpr[i], tpr[i], th[i] = roc_curve(Ytest[:, i], pred[j][:,i+1])
        roc_auc[i] = auc(fpr[i], tpr[i])
    J[j] = tpr[0] - fpr[0]
    thr[j] = th[0]
    plt.plot(
        fpr[0],
        tpr[0],
        color=predcolor[j],
        lw=lw,
        label="%s (AUC = %0.3f)" %(label[j], roc_auc[0])
    )
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel("1-spectivity")
plt.ylabel("sencitivity")
#plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
plt.savefig('D:/physionet.org/files/pic/XBGvsMLsBY.tif')

#本院数据处理
#load model
Xtrain = pd.read_csv('D:/physionet.org/files/Xtrain2.csv').iloc[:,1:]
Xtest = pd.read_csv('D:/physionet.org/files/Xtest2.csv').iloc[:,1:]
Ytrain = pd.read_csv('D:/physionet.org/files/Ytrain2.csv').iloc[:,1:].values
Ytest = pd.read_csv('D:/physionet.org/files/Ytest2.csv').iloc[:,1:].values
model.load_model('D:/physionet.org/files/model2.json')
patient_filePath_wfy='D:/physionet.org/files/本院脓毒症4_by.csv'
bydata = pd.read_csv(patient_filePath_wfy)
bydata.drop(["pid"],axis=1,inplace=True)
bydata.drop(["blid"],axis=1,inplace=True)
bydata.loc[bydata["glucose"]>0,"glucose"]=bydata.loc[bydata["glucose"]>0]["glucose"]*18;
bydata.loc[bydata["albumin"]>0,"albumin"]=bydata.loc[bydata["albumin"]>0]["albumin"]/10;
bydata.loc[bydata["creatinine"]>0,"creatinine"]=bydata.loc[bydata["creatinine"]>0]["creatinine"]/88.4;
bydata.loc[bydata["platelets"]>0,"platelets"]=bydata.loc[bydata["platelets"]>0]["platelets"]*1.5;

for i in range(len(bydata.columns)):
    columnname=bydata.columns[i]
    if columnname=="vaso_total" \
    or columnname=="vaso_24hour" \
    or columnname=="cad"\
    or columnname=="hypertension"\
    or columnname=="diabetes" \
    or columnname=="liver"\
    or columnname=="etoh" \
    or columnname=="renal" \
    or columnname=="smoker"\
    or columnname=="rrt":
        bydata.loc[:,bydata.columns[i]].fillna(0,inplace=True)
 
weight_male_median = newdata.loc[newdata["gender"]==1,"weight"].median()
weight_female_median = newdata.loc[newdata["gender"]==0,"weight"].median()
height_male_median = newdata.loc[newdata["gender"]==1,"height"].median()
height_female_median = newdata.loc[newdata["gender"]==0,"height"].median()

bydata.loc[bydata["gender"]==1,"weight"]=bydata.loc[bydata["gender"]==1,"weight"].fillna(weight_male_median)
bydata.loc[bydata["gender"]==0,"weight"]=bydata.loc[bydata["gender"]==0,"weight"].fillna(weight_female_median)
bydata.loc[bydata["gender"]==1,"height"].fillna(height_male_median,inplace=True)
bydata.loc[bydata["gender"]==0,"height"].fillna(height_female_median,inplace=True)    
ignore_col=["sofa","apache","gcs"]
for i in range(len(bydata.columns)):
    if bydata.columns[i]=="dieinhosp":
        continue
    if bydata.columns[i] in ignore_col:
        bydata.loc[:,bydata.columns[i]].fillna(bydata.loc[:,bydata.columns[i]].median(),inplace=True)     
        continue
    if bydata.columns[i]=="sapsii":
        bydata.loc[:,"apache"].fillna(bydata.loc[:,"apache"].median(),inplace=True)
    else:
        bydata.loc[:,bydata.columns[i]].fillna(newdata.loc[:,bydata.columns[i]].median(),inplace=True)     

bydata.loc[:,"pao2fio2ratio"]=bydata["po2"]/bydata["fio2"]*100


bydata_oasis = bydata.loc[:,["oasis"]]
bydata_gcs = bydata.loc[:,["gcs"]]
bydata_sofa = bydata.loc[:,["sofa"]]
bydata_sap = bydata.loc[:,["sapsii"]]
bydata.to_csv('D:/physionet.org/files/concepts/本院脓毒症4_filled.csv') 
bydata.drop(["apache"],axis=1,inplace=True)
bydata.loc[bydata["dieinhosp"]!=4,"dieinhosp"]=0
bydata.loc[bydata["dieinhosp"]==4,"dieinhosp"]=1
target_by = bydata.loc[:,"dieinhosp"].values.reshape(-1,1)
data_by=bydata.drop(["dieinhosp"],axis=1,inplace=True)
data_by=bydata.drop(["oasis"],axis=1,inplace=True)
data_by=bydata.drop(["sofa"],axis=1,inplace=True)
data_by=bydata.drop(["sapsii"],axis=1,inplace=True)
#本院数据预处理 end

data=bydata.rename(columns={"weight": "Weight", "gcs": "GCS"
                     ,"peep": "PEEP", "bun": "BUN"
                     ,"aniongap": "Aniongap", "tidal": "Tidal volume"
                     ,"alp": "ALP", "resp_rate": "Respiratory Rate"
                     ,"sbp": "SBP", "heart_rate": "Heart Rate"
                     ,"lactate": "Lactate", "platelets": "Platelets"
                     ,"fio2": "Fio2", "temperature": "Temperature"
                     ,"pt": "PT", "spo2": "Spo2"
                     , "ldh": "LDH","hypertension":"Hypertension"
                     ,"ph": "PH", "age": "Age","mbp":"MBP"
                     ,"glucose":"Glucose","dbp":"DBP"
                     ,"creatinine":"Creatinine","antibiotic":"Antibiotic","ptt":"PTT"
                     ,"albumin":"Albumin","pao2fio2ratio":"P/F","vent_24time":"24hTV"})

#温附一病人数据ROC绘图
dtest_by = xgb.DMatrix(data)
Ypred_by = model.predict(dtest_by)

Ypred_by_lr = lrl2.predict_proba(data)

X_svm=pd.concat([Xtrain,data],axis=0)
X_svm = StandardScaler().fit_transform(X_svm) 

X_svm = pd.DataFrame(X_svm)
Xtrain_svm = X_svm.loc[0:Xtrain.shape[0]-1]
Xtest_svm = X_svm.loc[Xtrain.shape[0]:]

Ypred_by_svm = svc.predict_proba(Xtest_svm)


Ypred_by_dnn = dnn.predict_proba(data)

fpr = dict()
tpr = dict()
thby = dict()
thrby = dict()
roc_auc = dict()
Jby = dict()

plt.figure(figsize=(6,6),dpi=300)
lw = 1
label=["XGB","LR","SVM","DNN"]
pred=[Ypred_by,Ypred_by_lr,Ypred_by_svm,Ypred_by_dnn]
predcolor=["red","darkorange","purple","blue"]
for j in range(len(label)):
    for i in range(1):
        fpr[i], tpr[i], thby[i] = roc_curve(target_by[:, i], pred[j][:,i+1])
        roc_auc[i] = auc(fpr[i], tpr[i])
    Jby[j] = tpr[0] - fpr[0]
    thrby[j] = thby[0]
    plt.plot(
        fpr[0],
        tpr[0],
        color=predcolor[j],
        lw=lw,
        label="%s (AUC = %0.3f)" %(label[j], roc_auc[0])
    )
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel("1-spectivity")
plt.ylabel("sencitivity")
#plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
plt.savefig('D:/physionet.org/files/pic/XBGvsMLsBY2.tif')
#plt.show()

#sap单因素预测

dtest = xgb.DMatrix(bydata_sap)
Ypred_by_sap = model_sap.predict(dtest)
#gcs单因素预测

dtest = xgb.DMatrix(bydata_gcs)
Ypred_by_gcs = model_gcs.predict(dtest)
#sofa单因素预测

dtest = xgb.DMatrix(bydata_sofa)
Ypred_by_sofa = model_sofa.predict(dtest)
#oasis单因素预测

dtest = xgb.DMatrix(bydata_oasis)
Ypred_by_oasis = model_oasis.predict(dtest)

fpr = dict()
tpr = dict()
roc_auc = dict()
plt.figure(figsize=(6,6),dpi=300)
lw = 1
label=["XGB","SAPSII","GCS","SOFA","OASIS"]

pred=[Ypred_by,Ypred_by_sap,Ypred_by_gcs,Ypred_by_sofa,Ypred_by_oasis]
predY=[target_by,target_by,target_by,target_by,target_by]
predcolor=["red","darkorange","purple","blue","darkgreen","darkred","brown"]
for j in range(len(pred)):
    for i in range(1):
        fpr[i], tpr[i], _ = roc_curve(predY[j][:, i], pred[j][:,i+1])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    plt.plot(
        fpr[0],
        tpr[0],
        color=predcolor[j],
        lw=lw,
        label="%s (AUC = %0.3f)" %(label[j], roc_auc[0])
    )
    #sns.regplot(x=fpr[0], y=tpr[0], ci=95)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel("1-Specificity")
plt.ylabel("Sensitivity")
#plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
plt.savefig('D:/physionet.org/files/pic/本院评分.tif')
plt.show()
from sklearn.metrics import confusion_matrix
def sensitivityCalc(Predictions, Labels):
    MCM = confusion_matrix(Labels, Predictions)
    fp_sum = MCM[0][0]# TP 预测为0实际为0
    fn_sum = MCM[0][1] # FP 预测为0实际为1

    tn_sum = MCM[1][0] # FN 预测为1实际为0
    tp_sum = MCM[1][1] # TN 预测为1实际为1

    # 这里加1e-6，防止 0/0的情况计算得到nan，即tp_sum和fn_sum同时为0的情况
    Condition_negative = tp_sum + fn_sum + 1e-6

    sensitivity = tp_sum / Condition_negative
  
    return sensitivity
def specificityCalc(Predictions, Labels):
    MCM = confusion_matrix(Labels, Predictions)
    fp_sum = MCM[0][0]# TP 预测为0实际为0
    fn_sum = MCM[0][1] # FP 预测为0实际为1

    tn_sum = MCM[1][0] # FN 预测为1实际为0
    tp_sum = MCM[1][1] # TN 预测为1实际为1

    Condition_negative = tn_sum + fp_sum + 1e-6

    Specificity = tn_sum / Condition_negative
    ppv = tp_sum/(tp_sum+fp_sum)
    npv = tn_sum/(tn_sum+fn_sum)
    return Specificity,ppv,npv
def argmax(array):
    val=-1000
    index=-1
    for i in range(len(array)):
        if val<array[i]:
            val=array[i]
            index=i
    return index
label=["XGB","LR","SVM","DNN","SAPSII","GCS","SOFA","OASIS"]
pred=[Ypred,Ypred_lr,Ypred_svm,Ypred_dnn,Ypred_sap,Ypred_gcs,Ypred_sofa,Ypred_oasis]

Ypredth =(Ypred[:,1]>thr[0][argmax(J[0])])+0
Ypred_lrth =(Ypred_lr[:,1]>thr[1][argmax(J[1])])+0
Ypred_svmth =(Ypred_svm[:,1]>thr[2][argmax(J[2])])+0
Ypred_dnnth =(Ypred_dnn[:,1]>thr[3][argmax(J[3])])+0
Ypred_sapth =(Ypred_sap[:,1]>thrpf[1][argmax(Jpf[1])])+0
Ypred_gcsth =(Ypred_gcs[:,1]>thrpf[2][argmax(Jpf[2])])+0
Ypred_sofath =(Ypred_sofa[:,1]>thrpf[3][argmax(Jpf[3])])+0
Ypred_oasisth =(Ypred_oasis[:,1]>thrpf[4][argmax(Jpf[4])])+0
predth=[Ypredth,Ypred_lrth,Ypred_svmth,Ypred_dnnth,Ypred_sapth,Ypred_gcsth,Ypred_sofath,Ypred_oasisth] 
index=[argmax(J[0]),argmax(J[1]),argmax(J[2]),argmax(J[3]),argmax(Jpf[1]),argmax(Jpf[2]),argmax(Jpf[3]),argmax(Jpf[4])]
predY=[Ytest,Ytest,Ytest,Ytest,Ytest_sap,Ytest_gcs,Ytest_sofa,Ytest_oasis]       
columns=["auc","se","sp","ac","f1_score","ppv","npv"]
report=pd.DataFrame(columns=columns);
for i in range(len(label)):
    df_row = report.shape[0]
    report.loc[df_row] = [0.000,0.000,0.000,0.000,0.000,0.000,0.000]
   
    fpr[i], tpr[i], _ = roc_curve(predY[i][:, 0], pred[i][:,1])
    roc_auc[i] = auc(fpr[i], tpr[i]) 
    report.loc[df_row]["ac"] = accuracy_score(predY[i][:, 0], predth[i])
    report.loc[df_row]["auc"] = roc_auc[i] 
    report.loc[df_row]["se"] = tpr[i][index[i]]

    report.loc[df_row]["sp"] = 1-fpr[i][index[i]]
    MCM = confusion_matrix(predY[i][:, 0], predth[i])
    tn_sum = MCM[0][0]# TP 预测为0实际为0
    fn_sum = MCM[0][1] # FP 预测为0实际为1

    fp_sum = MCM[1][0] # FN 预测为1实际为0
    tp_sum = MCM[1][1] # TN 预测为1实际为1
    report.loc[df_row]["ppv"] = tp_sum/(tp_sum + fp_sum)
    report.loc[df_row]["npv"] = tn_sum/(tn_sum + fn_sum)
    report.loc[df_row]["f1_score"] =f1_score(predY[i][:, 0], predth[i])







