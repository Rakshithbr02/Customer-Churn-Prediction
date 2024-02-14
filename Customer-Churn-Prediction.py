#!/usr/bin/env python
# coding: utf-8

# # importing necessary files 

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv('telecom_customer_churn.csv')


# In[3]:


df.head(5)


# # overviewing all the columns in the dataset

# In[4]:


df.columns


# In[5]:


df1 = df.copy()


# # Creating a copy of the Dataset

# In[6]:


df1.head(7)


# In[7]:


df1.columns


# # Explotratory Data Analysis

# ## Data Preprocessing

# ### Dropping unwanted columns from the Dataset

# In[9]:


df1.drop(['Customer ID','Total Refunds','Zip Code','Latitude', 'Longitude','Churn Category', 'Churn Reason'],axis='columns',inplace=True)


# In[10]:


df1.shape


# In[11]:


df1.dtypes


# ### Checking the number pf unique value in each column

# In[12]:


features = df1.columns
for feature in features:
     print(f'{feature}--->{df[feature].nunique()}')


# ### Getting the percentage of Null Values in each Column

# In[13]:


df1.isnull().sum() / df1.shape[0]


# ### Cleaning Functions for the Dataset

# In[14]:


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame)
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


# In[15]:


df1=df1.interpolate()


# In[16]:


df1=df1.dropna()
df.head()


# In[17]:


df['Unlimited Data'] 


# In[18]:


number_columns=['Age','Number of Dependents','Number of Referrals','Tenure in Months','Avg Monthly Long Distance Charges','Avg Monthly GB Download','Monthly Charge','Total Charges','Total Extra Data Charges','Total Long Distance Charges','Total Revenue']


# ### Checking the unique values of column having datatype : 'object'

# In[19]:


def unique_values_names(df):
    for column in df:
        if df[column].dtype=='object':
            print(f'{column}:{df[column].unique()}')


# In[20]:


unique_values_names(df1)


# # Data Visualization

# In[21]:


import plotly.express as px 


# ### Visualizing Column 'Age' in the dataset

# In[22]:


fig = px.histogram(df1, x = 'Age')
fig.show()


# ### Checking the stats in number_columns of the copied dataset

# In[23]:


df1.hist(figsize=(15,15), xrot=30)


# In[24]:


df1['Age']


# In[25]:


import matplotlib.pyplot as plt


# ### Visualizing the number of customers who churned, stayed or joined in the company with a bar plot

# In[26]:


Customer_Stayed=df1[df1['Customer Status']=='Stayed'].Age
Customer_Churned=df1[df1['Customer Status']=='Churned'].Age
Customer_Joined=df1[df1['Customer Status']=='Joined'].Age

plt.xlabel('Age')
plt.ylabel('Customers Numbers')
plt.hist([Customer_Stayed,Customer_Churned,Customer_Joined], color=['black','red','blue'],label=['Stayed','Churned','Joined'])

plt.title('Customers Behavior ',fontweight ="bold")
plt.legend()


# In[27]:


import seaborn as sns


# ### Defining Correlation between the columns in the dataset

# In[28]:


data  = df1.corr()
plt.figure(figsize = (20,10))
sns.heatmap(data, annot = True)


# ### Analyzing Outlier in the dataset with respect to customer status

# In[29]:


fig, ax = plt.subplots(4,3, figsize = (15,15))
for i, subplot in zip(number_columns, ax.flatten()):
    sns.boxplot(x = 'Customer Status', y = i , data = df1, ax = subplot)


# In[30]:


fig = px.density_heatmap(df1, x='Age', y='Total Charges')
fig.show()


# In[31]:


df1.columns


# In[32]:


pd.crosstab(df['Customer Status'], df['Married']).plot(kind='bar')


# In[33]:


pd.crosstab(df['Customer Status'], df['Gender']).plot(kind='bar')


# In[34]:


df1['Payment Method'].unique()


# ### Create dictionary with role / data key value pairs

# In[35]:


Roles = {}
for j in df1['Payment Method'].unique():
    Roles[j] = df1[df1['Payment Method'] == j]


# In[36]:


Roles.keys()


# ### Selecting the rows where the role is 'Credit Card'

# In[37]:


Roles['Credit Card']


# In[38]:


len(Roles)


# ### Checking the number of Offers in the dataset

# In[39]:


off = df1['Offer'].value_counts()
off


# In[40]:


import plotly.graph_objects as go


# In[41]:


fig = go.Figure([go.Bar(x=off.index, y=off.values)])
fig.show()


# In[42]:


df1_off = Roles['Credit Card'].Offer.value_counts()
df1_off


# In[43]:


fig = go.Figure([go.Bar(x= df1_off.index, y=df1_off.values)])
fig.show()


# In[44]:


df1 = df1.rename(columns = {'Customer Status':'Customer_Status'})


# In[45]:


Roles1 = {}
for k in df1['Customer_Status'].unique():
    Roles1[k] = df1[df1['Customer_Status'] == k]
Roles1.keys()


# In[46]:


df1_state = Roles1['Stayed'].Offer.value_counts()
df1_state


# # Data Modelling

# #### Replacing the Gender column in the dataset with Label Encoding
# 
# #### 0 for Female
# 
# #### 1 for Male

# In[48]:


df1.replace({"Gender":{'Female':0,'Male':1}},inplace=True)


# #### Replacing the columns with 'yes' and 'no' output by Label Encoding
# 
# #### 0 for No
# 
# #### 1 for Yes

# In[49]:


yes_and_no=[  'Paperless Billing', 'Unlimited Data', 
       'Streaming Movies', 'Streaming Music',  'Streaming TV',
       'Premium Tech Support', 'Device Protection Plan', 'Online Backup', 'Online Security',
       'Multiple Lines',  'Married']
for i in yes_and_no:
    df1.replace({'No':0,'Yes':1},inplace=True)


# #### Replacing 'Phone Service' with '1'

# In[59]:


df1.replace({"Phone Service":{'Yes':1}},inplace=True)


# In[64]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df1.Customer_Status = le.fit_transform(df1.Customer_Status)


# In[71]:


cols_to_scale = ['Age','Number of Dependents','Number of Referrals','Tenure in Months','Avg Monthly Long Distance Charges','Avg Monthly GB Download','Monthly Charge', 'Total Charges',
       'Total Extra Data Charges', 'Total Long Distance Charges','Total Revenue']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df1[cols_to_scale] = scaler.fit_transform(df1[cols_to_scale])


# ## Dealing with Imbalance Data

# #### Dropping the Customer_Status
# 
# #### i.e. The column tht we have to predict and set as a dependent variable

# In[72]:


X = df1.drop('Customer_Status',axis='columns')
y = df1['Customer_Status']


# In[73]:


X.head(5)


# In[74]:


y.head(5)


# # Data Model Building

# ### Spliiting the data in Training and Test Data

# In[75]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=5)


# In[76]:


len(X_train)


# In[77]:


X_train[:10]


# ## Importing the required files for the model that is to applied
# 
# #### 1. Random Forest Classifier
# #### 2. Logistic Regression
# #### 3. GaussianNB
# #### 4. Decision Tree Classifier
# #### 5. XGB Classifier
# 

# ## Impoting Models

# In[79]:


get_ipython().system('pip install xgboost')


# In[80]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


# In[81]:


model_params = {
     
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': [1,5,10]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'C': [1,5,10]
        }
    },
    'naive_bayes_gaussian': {
        'model': GaussianNB(),
        'params': {}
    },
    
    'decision_tree': {
        'model': DecisionTreeClassifier(),
        'params': {
            'criterion': ['gini','entropy'],
        }    
    },
       'XGB_Classifier':{
        'model':XGBClassifier(),
        'params':{
            'base_score':[0.5]
            
        }
    },   
}


# ##### It was concluded that XGB_Classifier was giving us the best_score in the dataset

# ## Selecting the model with best score for the dataset

# In[82]:


reg=XGBClassifier()
reg.fit(X_train, y_train)


# In[83]:


reg.score(X_test, y_test)


# ##### We got an accuracy of 80.86 percent in the testing dataset

# ### Predicting values from the model build to check the accuracy

# In[84]:


y_predicted = reg.predict(X_test)
y_predicted[:5]


# ### Verifying the actual values with the predicted values

# In[85]:


y_test[:5]


# ## Importing Confusion Matrx 

# In[86]:


import seaborn as sn
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# ### Importing Classification Report

# In[87]:


from sklearn.metrics import classification_report


# In[88]:


print(classification_report(y_test, y_predicted))


# In[89]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_predicted)


# ##### In the end we conclude that the Telecom Customer Churn Prediction was best worked with XGB_Classifier with an accuracy score of 80.86%
