#!/usr/bin/env python
# coding: utf-8

# # BANKING PROJECT

# In[1]:


# Importing data libraries
import pandas as pd
import numpy as np 
import os

# To display number rows and columns
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter("ignore")


# In[2]:


Banking_data=pd.read_excel("C:/ALEKYA/casptone/Project2_Dataset/Dataset/data.xlsx")


# In[3]:


Banking_data.head(2)


# In[4]:


Banking_data.shape


# In[5]:


print ("Shape of data: {}" . format (Banking_data.shape))
print ("Number of rows: {}" . format (Banking_data.shape [0]))
print ("Number of columns: {}" . format (Banking_data.shape [1]))


# In[6]:


#1. Perform preliminary data inspection and report the findings as the structure of the data, missing 
# values, duplicates, etc
Banking_data.dtypes


# In[7]:


Banking_data.info()


# In[8]:


Banking_data.nunique()


# In[9]:


Banking_data.isnull().sum()


# In[10]:


Banking_data.nunique()


# In[11]:


Banking_data.duplicated().sum()


# In[12]:


#  2. Variable names in the data may not be in accordance with the identifier naming in Python so, 
# change the variable names accordingly.
Banking_data.columns


# In[13]:


new_col=[]
for col_name in Banking_data.columns:
    new_col.append(str(col_name.replace('.',"_")))
print(new_col)


# In[14]:


Banking_data.columns=new_col
Banking_data.columns


# In[15]:


Banking_data.Employment_Type.value_counts()


# In[16]:


Banking_data.describe()['loan_default']


# In[17]:


#3. The presented data might also contain some missing values therefore, exploration will also lead 
# to devising strategies to fill in the missing values while exploring the data

Banking_data.Employment_Type.value_counts()


# In[18]:


Banking_data.loan_default.value_counts(normalize=True)*100


# In[19]:


Banking_data.Employment_Type.value_counts(normalize=True)*100


# In[20]:


Banking_data.Employment_Type.fillna('Self employed',inplace=True)
Banking_data.isnull().sum().sum()


# In[21]:


Banking_data.Employment_Type.value_counts()


# In[22]:


#4. Provide the statistical description of the quantitative data variables
Banking_data.describe()


# In[23]:


# 5. Explain how is the target variable distributed overall
sns.countplot(x='loan_default', data=Banking_data)
plt.title('Distribution of Loan Default')
plt.xlabel('Loan Default')
plt.ylabel('Count')
plt.show()


# In[24]:


#6. Study the distribution of the target variable across various categories like branch, city, state,
# branch, supplier, manufacturer, etc.
variable_list=['Employment_Type','State_ID','branch_id']

for i in variable_list:
    display(Banking_data.groupby([i])[['loan_default']].mean().sort_values('loan_default'))


# In[25]:


# 7. What are the different employment types given in the data? Can a strategy be developed to fill in
#the missing values (if any)? Use pie charts to express the different types of employment that
# define the defaulters and non-defaulters.
Banking_data.groupby('Employment_Type').size().plot(kind='pie',autopct='%.2f')


# In[26]:


Banking_data.head(2)


# In[27]:


# 8. Has age got anything to do with defaulting? What is the distribution of age w.r.t. to the
#defaulters and non-defaulters?
Banking_data['Person_Age']=2022-Banking_data['Date_of_Birth'].dt.year
Banking_data.head()


# In[28]:



Banking_data.groupby([pd.cut(Banking_data['Person_Age'],5)])['loan_default'].mean()


# In[29]:


# 9. What type of ID was presented by most of the customers for proof?
id_col=['Aadhar_flag','PAN_flag','VoterID_flag','Passport_flag']

for i in id_col:
    print('The number of people used the id',i,':', Banking_data[i].sum())


# In[30]:


#primary
Banking_data.AVERAGE_ACCT_AGE.value_counts(normalize=True)


# In[31]:



Banking_data['ACCT_Age_Bracket']=Banking_data['AVERAGE_ACCT_AGE'].apply(lambda x:str(x).split()[0])
Banking_data.head(1)


# In[32]:


Banking_data.groupby(['ACCT_Age_Bracket'])['loan_default'].mean().to_frame().sort_values('loan_default',ascending=False)


# In[33]:


Banking_data.describe()


# In[51]:


#  Study the credit bureau score distribution. Compare the distribution for defaulters vs. non
# defaulters. Explore in detail.
Banking_data.groupby([pd.cut(Banking_data['PRI_NO_OF_ACCTS'],5)])['loan_default'].mean()


# In[35]:


Banking_data.groupby([pd.cut(Banking_data['SEC_NO_OF_ACCTS'],5)])['loan_default'].mean()


# In[36]:


Banking_data.groupby([pd.cut(Banking_data['NO_OF_INQUIRIES'],2)])['loan_default'].mean()


# In[37]:


Banking_data.groupby([pd.cut(Banking_data['NEW_ACCTS_IN_LAST_SIX_MONTHS'],2)])['loan_default'].mean()


# In[38]:


Banking_data.groupby([pd.cut(Banking_data['DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS'],2)])['loan_default'].mean()


# # Performance of Logistic Regression

# In[39]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[40]:


Banking_data.columns


# In[41]:


Banking_data.head()


# In[42]:


x=Banking_data[['disbursed_amount', 'asset_cost', 'ltv', 'branch_id',
       'supplier_id', 'manufacturer_id', 'Current_pincode_ID','DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS','NO_OF_INQUIRIES',
    'Person_Age', 'PRI_NO_OF_ACCTS', 'PRI_ACTIVE_ACCTS',
       'PRI_OVERDUE_ACCTS', 'PRI_CURRENT_BALANCE', 'PRI_SANCTIONED_AMOUNT',
       'PRI_DISBURSED_AMOUNT', 'SEC_NO_OF_ACCTS', 'SEC_ACTIVE_ACCTS',
       'SEC_OVERDUE_ACCTS', 'SEC_CURRENT_BALANCE', 'SEC_SANCTIONED_AMOUNT',
       'SEC_DISBURSED_AMOUNT', 'PRIMARY_INSTAL_AMT', 'SEC_INSTAL_AMT']]#features
y=Banking_data['loan_default']


# In[43]:


y.value_counts()


# In[44]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)


# ### Sklearn Logic

# In[45]:


model=LogisticRegression()

model.fit(x_train,y_train)


# In[46]:


model.score(x_test,y_test)


# In[47]:


y_pred=model.predict(x_test)
y_pred


# In[48]:


from sklearn.metrics import confusion_matrix,classification_report


# In[49]:


confusion_matrix(y_test,y_pred)


# In[50]:


print(classification_report(y_test,y_pred))


# In[ ]:




