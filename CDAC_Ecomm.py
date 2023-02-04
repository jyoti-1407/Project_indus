#!/usr/bin/env python
# coding: utf-8

# # Ecommerce POC

# In[101]:


import pandas as pd
import numpy as np
import seaborn as sns
import math as ma
import matplotlib.pyplot as plt
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import PCA
from sklearn.metrics import jaccard_similarity_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from math import sqrt
from sklearn import tree
import scipy.stats as ss
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
get_ipython().run_line_magic('matplotlib', 'inline')


# In[102]:


# Reading CSV
Ecom_Data = pd.read_csv(r"C:\Venkat PRojects\ALL_PROJECTS\2022 Projects\Accenture\Accenture\Day5\07 Supervised  Learning\Ecommerce\Ecom.csv")


# In[103]:


Ecom_Data.head()


# In[ ]:





# In[104]:


Ecom_Data = pd.read_csv("C:\\Users\\Dell 5370\\Desktop\\IBM 18-22 March\\White papers\\Ecommerce Python\\Ecommerce.csv")


# In[105]:


# converting required variables into factors
#Since factor columns are already segregated fact1 to fact30.
print(Ecom_Data.head())
Ecom_Data.columns
Ecom_Data.shape


# In[106]:


# creating new data frame
Ec_Data = Ecom_Data
Ec_Data.isnull().sum()


# In[107]:


# creating another data set
Ey_Data = Ecom_Data
Ey_Data = Ey_Data.fillna(method='pad')
Ey_Data.isnull().sum()


# In[108]:


# Filling null values with forward or backword fill
Ec_Data = Ec_Data.fillna(method='pad')
Ec_Data.isnull().sum()
y = len(Ec_Data.columns)
Ey_Data = Ec_Data
y


# In[109]:


# New code # using label encoder & for loop converting to factors above defined columns
lb_make = LabelEncoder()
for i in range(y):
    Ey_Data[Ec_Data.columns[i]] = lb_make.fit_transform(Ey_Data[Ec_Data.columns[i]])
    i+=1
print("its done") 


# In[110]:


Ey_Data.head()
Ec_Data.columns


# In[111]:


Ec_Data.shape
Ec_Data.head()


# In[112]:


Ec_Data1= pd.DataFrame()


# In[113]:


# Applying PCA to factor variabls to reduce the dimensions
# creating data set only with factor elements
Ec_Data1 = Ec_Data[["FACT__1","FACT__2","FACT__3","FACT__4","FACT__5","FACT__6","FACT__7","FACT__8","FACT__9","FACT__10","FACT__11","FACT__12","FACT__13","FACT__14","FACT__15","FACT__16","FACT__17","FACT__18","FACT__19","FACT__20","FACT__21","FACT__22","FACT__23","FACT__24","FACT__25","FACT__26","FACT__27","FACT__28","FACT__29","FACT__30"]]
Ec_Data1.head()


# In[114]:


Ec_Data1.shape


# In[115]:


# checking cronbach alpha value to under satand reliability
def CronbachAlpha(itemscores):
    itemscores = np.asarray(itemscores)
    itemvars = itemscores.var(axis=1, ddof=1)
    tscores = itemscores.sum(axis=0)
    nitems = len(itemscores)

    return nitems / (nitems-1.) * (1 - itemvars.sum() / tscores.var(ddof=1))


# In[116]:


# CronbachAlpha Test
CronbachAlpha(Ec_Data)


# In[117]:


# CronbachAlpha Test
CronbachAlpha(Ey_Data)


# In[143]:


Ey_Data.columns


# In[144]:


#KMO Test for Data Quality
from factor_analyzer import FactorAnalyzer  as FA
#FA.calculate_kmo(Ey_Data)
df_features = Ey_Data
fa = FactorAnalyzer()
fa.analyze(df_features, 3, rotation=None)
fa.get_communalities()


# In[145]:


#factor_analyzer.factor_analyzer.calculate_kmo(pd.df_features)
#factor_analyzer.FactorAnalyzer.partial_correlations(pd.df_features)
factor_analyzer.factor_analyzer.FactorAnalyzer.calculate_kmo(pd.df_features)


# In[146]:


# creating covariance matrix
CVM = PCA(n_components=30)
# calculating eigen values
CVM.fit(Ec_Data1)


# In[122]:


#calculate variance ratios
variance = CVM.explained_variance_ratio_
#cumulative sum of variance explained with [n] features
var=np.cumsum(np.round(variance, decimals=3)*100)
var


# In[123]:


# fitting PCA components to a data frame
pca_CVM = CVM.fit_transform(Ec_Data1)
pca_df = pd.DataFrame(data=pca_CVM,columns = ['pc1','pc2','pc3','pc4','pc5','pc6','pc7','pc8','pc9','pc10','pc11','pc12','pc13','pc14','pc15','pc16','pc17','pc18','pc19','pc20','pc21','pc22','pc23','pc24','pc25','pc26','pc27','pc28','pc29','pc30'])
pca_df.columns


# In[124]:


# Re running PCA with required columns
Req_Df =  pd.DataFrame()
Req_Df = pca_df.drop(['pc09', 'pc17', 'pc18', 'pc19','pc20', 'pc21', 'pc22', 'pc23', 'pc24', 'pc25', 'pc26', 'pc27', 'pc28',
                      'pc29', 'pc30'],axis=1)
Req_Df.head()


# In[125]:


from sklearn.metrics import confusion_matrix,classification_report


# In[126]:


# shaping data frame without demograhic & pca comopents
EC_NoDemo = Ey_Data.drop(['Gender','Marital status','age group','Educatuon','income','Job','Area',"FACT__1","FACT__2","FACT__3","FACT__4","FACT__5","FACT__6","FACT__7","FACT__8","FACT__9","FACT__10","FACT__11","FACT__12","FACT__13","FACT__14","FACT__15","FACT__16","FACT__17","FACT__18","FACT__19","FACT__20","FACT__21","FACT__22","FACT__23","FACT__24","FACT__25","FACT__26","FACT__27","FACT__28","FACT__29","FACT__30"],axis=1)
EC_NoDemo.columns


# In[127]:


# creating data frame with PCA components
Ec_New = pd.concat([Req_Df,EC_NoDemo],axis=1)
Ec_New.head()


# # Model For Gender

# # PCA Visualization  Boxplot Histograms 
# 

# In[128]:


Ec_New['pc2'].hist()
Ec_New['pc3'].hist()
Ec_New['pc4'].hist()
Ec_New['pc5'].hist()


# In[129]:


Ec_New['pc12'].hist()
Ec_New['pc13'].hist()
Ec_New['pc14'].hist()
Ec_New['pc15'].hist()


# In[130]:


Ec_New['pc7'].hist()
Ec_New['pc8'].hist()
Ec_New['pc9'].hist()
Ec_New['pc10'].hist()


# In[131]:


Ecom_Data.columns


# In[132]:


# Adding required demographic variable & build the model in Ec_Data1
Ec_Gender = pd.concat([Ec_New,Ey_Data[['Gender']]],axis=1)


# In[133]:


# Bulding model
X= Ec_Gender[['CBB_9','CBB_10','CBB_11','CBB_12','CBB_13','CBB_14','pc1','pc2','pc3','pc4','pc5','pc6','pc7','pc8','pc9','pc10','pc11','pc12','pc13','pc14','pc15']]
y=Ec_Gender['Gender']


# In[134]:



from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4, random_state=42)


# In[135]:


X.head()


# In[136]:


y.head()


# # Logistic for Gender

# In[137]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[138]:


predictions =  logmodel.predict(X_test)
print(predictions)


# In[139]:


# printing predictions
print(classification_report(y_test,predictions))
print("confuison matrix")
print(confusion_matrix(y_test,predictions))


# In[140]:


# computing ROC curve
fpr, tpr, _ = roc_curve(y_test,predictions)
plt.clf()
plt.plot(fpr,tpr)
plt.show()


# In[141]:


# Roc curve score
roc_auc_score(y_test,predictions)


# In[142]:


auc(fpr,tpr)


# In[57]:


# model validation
cohen_kappa_score(y_test, predictions)


# # Decision tree For Gender

# In[58]:


# Decision tree
clf = tree.DecisionTreeClassifier()
clf.fit(X_train,y_train)


# In[59]:


Dpredict = clf.predict(X_test)
print(classification_report(y_test,Dpredict))
print(confusion_matrix(y_test,Dpredict))


# In[60]:


cohen_kappa_score(y_test,Dpredict)


# # Navie bisen for Gender

# In[61]:


from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
Naive_Gender = GaussianNB()
# multiple variables
multiNaive = MultinomialNB()
Naive_Gender.fit(X_train, y_train)


# In[62]:


Npredict = Naive_Gender.predict(X_test)


# In[63]:


# printing predictions
print(classification_report(y_test,Npredict))
print("confuison matrix")
print(confusion_matrix(y_test,Npredict))


# # KNN model for Gender

# In[64]:


from sklearn.neighbors import KNeighborsClassifier


# In[65]:


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)


# In[66]:


pred = knn.predict(X_test)


# In[67]:


# printing predictions
print(classification_report(y_test,pred))
print("confuison matrix")
print(confusion_matrix(y_test,pred))


# # Logistic model for Age

# In[68]:


Ec_Age = pd.concat([Ec_New,Ey_Data[['age group']]],axis=1)


# In[69]:


# Training and Testing
X1= Ec_Age[['CBB_9','CBB_10','CBB_11','CBB_12','CBB_13','CBB_14','pc1','pc2','pc3','pc4','pc5','pc6','pc7','pc8','pc9','pc10','pc11','pc12','pc13','pc14','pc15']]
y1=Ec_Age['age group']
X1_train,X1_test,y1_train,y1_test =  train_test_split(X1,y1,test_size=0.3,random_state=101)


# In[70]:


# Bulding model
logmodel = LogisticRegression()
logmodel.fit(X1_train,y1_train)


# In[71]:


predictions =  logmodel.predict(X1_test)


# In[72]:


# printing predictions
print(classification_report(y1_test,predictions))
print("confuison matrix")
print(confusion_matrix(y1_test,predictions))


# # Decision tree for Age

# In[73]:


# Decision tree
clf = tree.DecisionTreeClassifier()
clf.fit(X1_train,y1_train)


# In[74]:


Dpredict = clf.predict(X1_test)
print(classification_report(y1_test,Dpredict))
print("confusion matrix")
print(confusion_matrix(y1_test,Dpredict))


# # KNN model for Age

# In[75]:


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X1_train,y1_train)


# In[76]:


pred = knn.predict(X1_test)


# In[77]:


# printing predictions
print(classification_report(y1_test,pred))
print("confuison matrix")
print(confusion_matrix(y1_test,pred))


# # Navie bisen for Age - Not worked for Age

# In[78]:


#Naive_Age = GaussianNB()
multiNaive = MultinomialNB()
multiNaive.fit(X1_train, y1_train)


# In[79]:


Npredict = multiNaive.predict(X1_test)
print(classification_report(y1_test,Npredict))
print("confuison matrix")
print(confusion_matrix(y1_test,Npredict))


# # Marital status

# In[80]:


Ec_Marital = pd.concat([Ec_New,Ey_Data[['Marital status']]],axis=1)


# In[81]:


Xm= Ec_Marital[['CBB_9','CBB_10','CBB_11','CBB_12','CBB_13','CBB_14','pc1','pc2','pc3','pc4','pc5','pc6','pc7','pc8','pc9','pc10','pc11','pc12','pc13','pc14','pc15']]
ym=Ec_Marital['Marital status']
Xm_train,Xm_test,ym_train,ym_test =  train_test_split(Xm,ym,test_size=0.3,random_state=101)


# # logistic Model

# In[82]:


logmodel = LogisticRegression()
logmodel.fit(Xm_train,ym_train)


# In[83]:


predictions =  logmodel.predict(Xm_test)


# In[84]:


print(classification_report(ym_test,predictions))
print("confuison matrix")
print(confusion_matrix(ym_test,predictions))


# # Decision Tree Martial status

# In[85]:


#  Decision tree
clf = tree.DecisionTreeClassifier()
clf.fit(Xm_train,ym_train)


# In[86]:


Dpredict = clf.predict(Xm_test)
print(classification_report(ym_test,Dpredict))
print("confusion matrix")
print(confusion_matrix(ym_test,Dpredict))


# # KNN model for Martial status

# In[87]:


# Knn model
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(Xm_train,ym_train)


# In[88]:


pred = knn.predict(Xm_test)


# In[89]:


# printing predictions
print(classification_report(ym_test,pred))
print("confuison matrix")
print(confusion_matrix(ym_test,pred))


# # Navie bisen for Martial Status

# In[90]:


Naive_NB = GaussianNB()
#multiNaive = MultinomialNB()
Naive_NB.fit(Xm_train, ym_train)


# In[91]:


Npredict = Naive_NB.predict(Xm_test)


# In[92]:


print(classification_report(ym_test,Npredict))
print("confuison matrix")
print(confusion_matrix(ym_test,Npredict))


# # Area Model

# In[93]:


Ec_Area = pd.concat([Ec_New,Ey_Data[['Area']]],axis=1)


# In[94]:


Xa= Ec_Area[['CBB_9','CBB_10','CBB_11','CBB_12','CBB_13','CBB_14','pc1','pc2','pc3','pc4','pc5','pc6','pc7','pc8','pc9','pc10','pc11','pc12','pc13','pc14','pc15']]
ya=Ec_Area['Area']
Xa_train,Xa_test,ya_train,ya_test =  train_test_split(Xa,ya,test_size=0.3,random_state=101)


# In[95]:


logmodel = LogisticRegression()
logmodel.fit(Xa_train,ya_train)


# In[96]:


predictions =  logmodel.predict(Xa_test)


# In[97]:


print(classification_report(ya_test,predictions))
print("confuison matrix")
print(confusion_matrix(ya_test,predictions))


# In[98]:


Ec_Area['Area'].tail()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




