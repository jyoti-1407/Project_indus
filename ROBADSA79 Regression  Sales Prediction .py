#!/usr/bin/env python
# coding: utf-8

# # Importing the packages

# In[1]:


import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt 
from scipy.stats import zscore
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn import linear_model
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')


# # Collecting the sales data

# In[2]:



sales=pd.read_csv("C:\\Users\\Dell 5370\\Desktop\\0Data Science  Programme Material\\Data Science Case Studies\\8 Linear Regression\\REgression Models  Python\\Sales_SONY.csv")


# In[3]:


sales.shape


# In[4]:


sales.corr()


# In[5]:


sales.head()


# In[6]:


sales.duplicated().any()


# In[7]:


sales.columns


# In[8]:


len(sales)


# In[9]:


sales.describe()


# # I Model_1 Sales with Advt

# we are building the model only on advt as per client requirment future 4Q sales predictions 
# budget allocating
# 2020 june 17L 
# 2020 sep 11L
# 2020 dec 9L
# 2021 march 16L

# # 1.1 Model_1(direct)

# In[10]:


import statsmodels.api as sm
from statsmodels.formula.api import ols
model=ols('Sales~Advt',data=sales).fit()
model1=sm.stats.anova_lm(model)
model1
print(model.summary())


# In[11]:


pre1=model.predict()
pre1
model.f_pvalue


# In[16]:


res1=sales['Sales'].values-pre1


# In[18]:


pre_1=pd.DataFrame(pre1,columns=["pre1"])
pre_1.head()


# In[19]:


res_1=pd.DataFrame(res1,columns=["res1"])
res_1.head()


# In[21]:


zscore1=pd.DataFrame(zscore(res1),columns=['zscore1'])
zscore1.head()


# In[22]:


sales1=pd.concat([sales,pre_1,res_1,zscore1],axis=1)
sales1
sales_1=pd.DataFrame(sales1)
sales_1.head()


# In[23]:


zscore1[zscore1['zscore1']>1.96]


# In[24]:


zscore1[zscore1['zscore1']<-1.96]


# # 1.2 Model_1 Applying (Dummy)

# We are applying dummy,where the value is above 1.96 as 1 and below 1.96 as 0 because those are the outliers to improve the model dummy variable is used.

# In[26]:


a=sales_1.copy()
for i in range(0,len(a)):
    if(np.any(a['zscore1'].values[i]>1.96)):
        a['zscore1'].values[i]=1
    else:
        a['zscore1'].values[i]=0
        test=a['zscore1']
        test
sales_1['dummy']=test
sales_1.tail()


# In[27]:


x=sales_1[["Advt","dummy"]]
y=sales_1["Sales"]
y.head()


# In[31]:


plt.scatter(y,res1)
plt.xlabel("res_adv")
plt.ylabel("Sales")


# In[33]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20,random_state=0)


# In[35]:


x_train.head()


# In[36]:


y_train.head()


# In[37]:


import statsmodels.api as sm
from statsmodels.formula.api import ols
x_train1=sm.add_constant(x_train)
model=sm.OLS( y_train,x_train1).fit()
print(model.summary())
ols = linear_model.LinearRegression()


# In[42]:


regr = linear_model.LinearRegression()
regr.fit(x_train,y_train)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)


# In[45]:


y_pred = regr.predict(x_test)
y_pred


# In[46]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[ ]:





# In[ ]:





# # 1.3 Model_1 Applying(Square)

# In[47]:


sales_1["sqr_Advt"]=sales_1["Advt"]**2
sales_1


# In[48]:


x=sales_1[["Advt","sqr_Advt"]]
y=sales_1["Sales"]
y.head()


# In[49]:


plt.scatter(y,res1)
plt.xlabel("res_adv")
plt.ylabel("Sales")


# In[50]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20,random_state=0)


# In[51]:


x_train


# In[52]:


y_train


# In[53]:


x_test


# In[54]:


y_test


# In[55]:


import statsmodels.api as sm
from statsmodels.formula.api import ols
x_train1=sm.add_constant(x_train)
model=sm.OLS( y_train,x_train1).fit()
print(model.summary())
ols = linear_model.LinearRegression()


# In[56]:


regr = linear_model.LinearRegression()
regr.fit(x_train,y_train)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)


# In[57]:


y_pred = regr.predict(x_test)
y_pred


# In[58]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# # 1.4 Model_1 Applying(Square root)

# In[59]:


sales_1["squareRoot_Advt"]=sales1["Advt"]**(1/2)
sales_1


# In[60]:


x_ADVT=sales_1[["Advt","squareRoot_Advt"]]
y_ADVT=sales_1["Sales"]
y_ADVT


# In[61]:


x_train, x_test, y_train, y_test = train_test_split(x_ADVT, y_ADVT, test_size=0.20,random_state=0)


# In[62]:


x_train


# In[63]:


y_train


# In[64]:


x_test


# In[65]:


y_test


# In[66]:


import statsmodels.api as sm
from statsmodels.formula.api import ols
x_train1=sm.add_constant(x_train)
model=sm.OLS( y_train,x_train1).fit()
print(model.summary())
ols = linear_model.LinearRegression()


# In[67]:


regr = linear_model.LinearRegression()
regr.fit(x_train,y_train)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)


# In[68]:


y_pred = regr.predict(x_test)
y_pred


# In[69]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# # 1.5 Model_1 Applying(log10)

# In[70]:


sales_1['log_Advt'] = np.log10(sales_1['Advt'])
sales_1   


# In[71]:


x_advt=sales_1[["Advt","log_Advt"]]
y_advt=sales_1["Sales"]
y_advt


# In[72]:


x_train, x_test, y_train, y_test = train_test_split(x_advt, y_advt, test_size=0.20,random_state=0)


# In[73]:


x_train


# In[74]:


y_train


# In[75]:


x_test


# In[76]:


y_test


# In[77]:


import statsmodels.api as sm
from statsmodels.formula.api import ols
x_train1=sm.add_constant(x_train)
model=sm.OLS( y_train,x_train1).fit()
print(model.summary())
ols = linear_model.LinearRegression()


# In[78]:


regr = linear_model.LinearRegression()
regr.fit(x_train,y_train)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)


# In[79]:


y_pred = regr.predict(x_test)
y_pred


# In[80]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# # II Model_2 Sales with PC
# 

# we are building the model only on PC as per client requirment future 4Q sales predictions budget allocating 2020 june 17L 2020 sep 11L 2020 dec 9L 2021 march 16L

# # 2.1 Model_2(direct)

# In[81]:


import statsmodels.api as sm
from statsmodels.formula.api import ols
model=ols('Sales~PC',data=sales).fit()
model2=sm.stats.anova_lm(model)
model2
print(model.summary())


# In[82]:


pre2=model.predict()
pre2


# In[83]:


pre_2=pd.DataFrame(pre2,columns=['pre2'])
pre_2


# In[84]:


res2=sales['Sales'].values-pre2
res2


# In[85]:


res_2=pd.DataFrame(res2,columns=['res2'])
res_2


# In[86]:


zscore2=pd.DataFrame(zscore(res2),columns=['zscore2'])
zscore2


# In[87]:


sales2=pd.concat([sales,pre_2,res_2,zscore2],axis=1)
sales2
sales_2=pd.DataFrame(sales2)
sales_2


# In[88]:


zscore2[zscore2['zscore2']>1.96]


# In[89]:


zscore2[zscore2['zscore2']<-1.96]


# In[90]:


b=sales_2.copy()
for i in range(0,len(b)):
    if(np.any(b['zscore2'].values[i]>1.96)):
        b['zscore2'].values[i]=0
    else:
        b['zscore2'].values[i]=1         
        test=b['zscore2']
        test
sales_2['dummy']=test
sales_2


# # 2.2 Model_2 Applying(Square)

# In[91]:


sales_2["sqr_PC"]=sales_2["PC"]**2
sales_2


# In[92]:


x_PC=sales_2[["PC","sqr_PC"]]
y_PC=sales_2["Sales"]
x_PC


# In[93]:


plt.scatter(y_PC,res2)
plt.xlabel("res_pc")
plt.ylabel("Sales")


# In[94]:


x_train, x_test, y_train, y_test = train_test_split(x_PC, y_PC, test_size=0.20,random_state=0)


# In[95]:


x_train


# In[96]:


y_train


# In[97]:


x_test


# In[98]:


y_test


# In[99]:


import statsmodels.api as sm
from statsmodels.formula.api import ols
x_train1=sm.add_constant(x_train)
model=sm.OLS( y_train,x_train1).fit()
print(model.summary())
ols = linear_model.LinearRegression()


# In[100]:


regr = linear_model.LinearRegression()
regr.fit(x_train,y_train)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)


# In[101]:


y_pred = regr.predict(x_test)
y_pred


# In[102]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# # 2.4 Model_2 Applying(SquareRoot)

# In[103]:


sales_2["squareRoot_PC"]=sales_2["PC"]**(1/2)
sales_2


# In[104]:


x_PC=sales_2[["PC","squareRoot_PC"]]
y_PC=sales_2["Sales"]
y_PC


# In[105]:


x_train, x_test, y_train, y_test = train_test_split(x_PC, y_PC, test_size=0.20,random_state=0)


# In[106]:


x_train


# In[107]:


y_train


# In[108]:


x_test


# In[109]:


y_test


# In[110]:


import statsmodels.api as sm
from statsmodels.formula.api import ols
x_train1=sm.add_constant(x_train)
model=sm.OLS( y_train,x_train1).fit()
print(model.summary())
ols = linear_model.LinearRegression()


# In[111]:


regr = linear_model.LinearRegression()
regr.fit(x_train,y_train)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)


# In[112]:


y_pred = regr.predict(x_test)
y_pred


# In[113]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# # 2.5 Model_2 Applying(Log10)

# In[114]:


sales_2['log_PC'] = np.log10(sales_2['PC'])
(sales_2)          


# In[115]:


x_pc=sales_2[["PC","log_PC"]]
y_pc=sales_2["Sales"]
y_pc


# In[116]:


x_train, x_test, y_train, y_test = train_test_split(x_pc, y_pc, test_size=0.20,random_state=0)


# In[117]:


x_train


# In[118]:


y_train


# In[119]:


x_test


# In[120]:


y_test


# In[121]:


import statsmodels.api as sm
from statsmodels.formula.api import ols
x_train1=sm.add_constant(x_train)
model=sm.OLS( y_train,x_train1).fit()
print(model.summary())
ols = linear_model.LinearRegression()


# In[122]:


regr = linear_model.LinearRegression()
regr.fit(x_train,y_train)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)


# y_pred = regr.predict(x_test)
# y_pred

# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# # III Model_3 Sales with Advt and PC 

# we are building the model only on Advt and PC as per client requirment future 4Q sales predictions budget allocating 2020 june 17L 2020 sep 11L 2020 dec 9L 2021 march 16L

# # 3.1 Model_3(direct)

# In[ ]:


import statsmodels.api as sm
from statsmodels.formula.api import ols
model=ols('Sales~Advt+PC',data=sales).fit()
model3=sm.stats.anova_lm(model)
model3
print(model.summary())


# In[ ]:


pre3=model.predict()
pre3


# In[ ]:


pre_3=pd.DataFrame(pre3,columns=['pre3'])
pre_3


# In[ ]:


res3=sales['Sales'].values-pre3
res3


# In[ ]:


res_3=pd.DataFrame(res3,columns=['res3'])
res_3


# In[ ]:


zscore3=pd.DataFrame(zscore(res3),columns=['zscore3'])
zscore3


# In[ ]:


zscore3[zscore3['zscore3']>1.96]


# In[ ]:


zscore3[zscore3['zscore3']<-1.96]


# In[ ]:


sales3=pd.concat([sales,pre_3,res_3,zscore3],axis=1)
sales3
sales_3=pd.DataFrame(sales3)
sales_3


# # 3.2 Model_3 Applying(Dummy)

# In[ ]:


c=sales_3.copy()
for i in range(0,len(c)):
    if(np.any(c['zscore3'].values[i]<-1.96)):
        c['zscore3'].values[i]=0
    else:
        c['zscore3'].values[i]=1         
        test=c['zscore3']
        test
sales_3['dummy']=test
sales_3


# # 3.3 Model_3 Applying(Square)

# In[ ]:


sales_3["sqr_pc"]=sales_3["PC"]**2
sales_3


# In[ ]:


x_adpc =sales_3[["Advt","PC","sqr_pc"]]
y_adpc = sales_3['Sales']
y_adpc


# In[ ]:


plt.scatter(y_adpc,res3)
plt.xlabel("res_adpc")
plt.ylabel("Sales")


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x_adpc,y_adpc,test_size=0.20,random_state=0)


# In[ ]:


x_train


# In[ ]:


y_train


# In[ ]:


x_test


# In[ ]:


y_test


# In[ ]:


import statsmodels.api as sm
from statsmodels.formula.api import ols
x_train1=sm.add_constant(x_train)
model=sm.OLS( y_train,x_train1).fit()
print(model.summary())
ols = linear_model.LinearRegression()


# In[ ]:


regr = linear_model.LinearRegression()
regr.fit(x_train,y_train)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)


# In[ ]:


y_pred = regr.predict(x_test)
y_pred


# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# # 3.4 Model_3 Applying(Square Root)

# In[ ]:


sales_3["squareRoot_pc"]=sales_3["PC"]**(1/2)
sales_3


# In[ ]:


x_adpc=sales_3[["Advt","PC","squareRoot_pc"]]
y_adpc=sales_3["Sales"]
y_adpc


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x_adpc,y_adpc,test_size=0.20,random_state=0)


# In[ ]:


x_train


# In[ ]:


y_train


# In[ ]:


x_test


# In[ ]:


y_test


# In[ ]:


import statsmodels.api as sm
from statsmodels.formula.api import ols
x_train1=sm.add_constant(x_train)
model=sm.OLS( y_train,x_train1).fit()
print(model.summary())
ols = linear_model.LinearRegression()


# In[ ]:


regr = linear_model.LinearRegression()
regr.fit(x_train,y_train)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)


# In[ ]:


y_pred = regr.predict(x_test)
y_pred


# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# # 3.5 Model_3 Applying(Log10)

# In[ ]:


sales_3['log_PC'] = np.log(sales_3['PC'])
(sales_3)


# In[ ]:


x_ADPC =sales_3[["Advt","PC","log_PC"]]
y_ADPC = sales_3['Sales']
x_ADPC


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x_ADPC,y_ADPC,test_size=0.20,random_state=0)


# In[ ]:


x_train


# In[ ]:


y_train


# In[ ]:


x_test


# In[ ]:


y_test


# In[ ]:


import statsmodels.api as sm
from statsmodels.formula.api import ols
x_train1=sm.add_constant(x_train)
model=sm.OLS( y_train,x_train1).fit()
print(model.summary())
ols = linear_model.LinearRegression()


# In[ ]:


regr = linear_model.LinearRegression()
regr.fit(x_train,y_train)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)


# In[ ]:


y_pred = regr.predict(x_test)
y_pred


# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[ ]:


x_test
y_test


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




