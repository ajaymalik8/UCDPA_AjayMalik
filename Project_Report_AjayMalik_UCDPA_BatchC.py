#!/usr/bin/env python
# coding: utf-8

# # Heart Attack Analysis & Prediction

# ## Install Required Libs ( If not installed already) 

# In[ ]:


#!pip install pandas
#!pip install NumPy


# In[ ]:


#!pip install matplotlib
#!pip install seaborn


# In[ ]:


#!pip install sklearn


# In[ ]:


#!pip install xgboost


# In[ ]:


#!pip install yellowbrick


# ## Import Required Lib

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report,roc_curve,RocCurveDisplay
from sklearn.metrics import recall_score, accuracy_score,roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


# In[2]:


# Apply the default theme
sns.set_theme()
sns.set_style("whitegrid")


# In[3]:


# Plot should appear inside the jupyter notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


plt.rcParams["figure.figsize"] = (10,6)
plt.rcParams['figure.dpi'] = 100


# In[5]:


pd.options.display.max_rows = 10
pd.options.display.max_columns = 40


# In[6]:


import warnings
warnings.filterwarnings("ignore")


# ## Importing Data for analysis

# In[7]:


ds_heart = pd.read_csv("Data/heart.csv")


# In[8]:


#Printing Dataset Shape
print("\nDateset Shape is : ",ds_heart.shape)


# ### Let's Understand Our Data

# In[9]:


ds_heart.sample()


# In[10]:


ds_heart.describe()


# In[11]:


#Columns List
ds_heart.columns


# In[12]:


#List of Numeric Columns
numeric_columns = [column for column in ds_heart.columns if (ds_heart[column].dtype == 'float64' or ds_heart[column].dtype == 'int64')]
print(numeric_columns)


# In[13]:


#Duplicate Values 
ds_heart.duplicated().sum()


# ### Data Cleaning and Manupulation using Function & RegEx
# 

# In[14]:


#Removing duplicate value 
ds_heart.drop_duplicates(inplace=True)


# In[15]:


#Total Records 303 unique records 302
#Printing Dataset Shape
print("\nDateset Shape is : ",ds_heart.shape,"(Unqiue Records)\n")


# In[16]:


ds_heart.columns


# In[17]:


## Rename few columns to understand 
ds_heart.rename(columns={'output': 'attack', 
                         'thall': 'stresstest',
                         'caa':'numberofmajorvessels',
                         'cp':'chestpaintype',
                         'exng':'exerciseinducedangina',
                         'restecg':'restingecg',
                         'fbs':'fastingbloodsugar',
                         'trtbps':'restingbloodpressure',
                         'thalachh':'maxheartrateachieved',
                         'slp':'slope',
                         'chol':'cholestoral'}, inplace=True) 
ds_heart.columns


# In[18]:


#finding Missing Values 
pd.options.display.max_rows = 15
print(ds_heart.isnull().sum())
pd.options.display.max_rows = 5


# In[19]:


#chest pain type: chest pain type
# 0: typical angina
# 1: atypical angina
# 2: non-anginal pain
# 3: asymptomatic
#Validating Values  
ds_heart.groupby(['chestpaintype'])['chestpaintype'].count()


# In[20]:


#fasting blood sugar > 120 mg/dl
#1 = true;
#0 = false
#Validating Values
ds_heart.groupby(['fastingbloodsugar'])['fastingbloodsugar'].count()


# In[21]:


# Heart Attack Count out of 303
#0 = No Hard Attack
#1 = Had Heart Attack 
#Validating Values
ds_heart.groupby(['attack'])['attack'].count()
# 164 Had Heart Attack out of 303


# In[22]:


#Thalium Stress Test result ~ (0,3) 
# 0 : Normal blood flow
# 1 : Abnormal blood flow during exercise - coronary artery disease
# 2 : Low blood flow during both rest and exercise -  severe blockage
# 3 : No thallium visible in parts of the heart - sign of damage from a heart attack
ds_heart.groupby(['stresstest'])['stresstest'].count()


# In[23]:


# Number of major vessels 
# Valid Values 0,1,2,3
ds_heart.groupby(['numberofmajorvessels'])['numberofmajorvessels'].count()


# In[24]:


# Found 4 Invalid record for Number of major vessels 
# Either Defaulting with meanvalue , max values or 
# Removing record with Invalid Values 
# To demostrate the concept of filling Missing Value we will use option 1 


# In[25]:


#Code to remove Invalid values (Not used)
#ds_heart=ds_heart[ds_heart.numberofmajorvessels!=4]


# In[26]:


ds_heart['numberofmajorvessels'] = ds_heart['numberofmajorvessels'].replace(4,np.nan)


# In[27]:


np.unique(ds_heart['numberofmajorvessels'])


# In[28]:


ds_heart['numberofmajorvessels'] = ds_heart['numberofmajorvessels'].fillna(ds_heart['numberofmajorvessels'].max())


# In[29]:


ds_heart = ds_heart.astype({'numberofmajorvessels':'int64'})
ds_heart.groupby(['numberofmajorvessels'])['numberofmajorvessels'].count()


# In[30]:


ds_heart.shape


# In[31]:


#Exercise induced angina 
# 1 = Yes, 
# 0 = No,
# Validating
ds_heart.groupby(['exerciseinducedangina'])['exerciseinducedangina'].count()


# In[32]:


#Resting electrocardiographic results 
# 0 = Normal
# 1 = ST-T wave normality
# 2 = Left ventricular hypertrophy
# Validating
ds_heart.groupby(['restingecg'])['restingecg'].count()


# In[33]:


#Max Heart Rate Achieved
#ds_heart.groupby(['maxheartrateachieved'])['maxheartrateachieved'].count()
np.unique(ds_heart['maxheartrateachieved'])


# In[34]:


#Resting Blood Pressure 
# 0 : downsloping
# 1 : flat
# 2 : upsloping
ds_heart.groupby(['slope'])['slope'].count()


# In[35]:


#cholestoral < 200 means healthy individual
#ds_heart.groupby(['cholestoral'])['cholestoral'].count()
np.unique(ds_heart['cholestoral'])


# ## Generating insights

# In[36]:


ds_attack = ds_heart[ds_heart.attack==1]


# In[37]:


ax= sns.distplot(ds_attack.age, rug=True, color="maroon")
plt.title("The distribution of ages")
plt.show()


# ## Age V/s Probability of Heart Attack

# In[38]:


def graphanalysis(x,y,z):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 1, 1)
    plt.title(x)
    sns.kdeplot(data=ds_heart[ds_heart.attack==1], x=y, hue = z, shade=True, palette="crest", ax=ax1, alpha=.3)
    sns.kdeplot(data=ds_heart[ds_heart.attack==0], x=y, shade=True, ax=ax1, palette="crest",alpha=0.3)
    plt.show()


# In[39]:


graphanalysis("Age & Heart attack relation","age","attack")


# ## Gender V/s Probability of Heart Attack

# In[40]:


ax= sns.countplot(ds_heart.sex, hue=ds_heart.attack, palette="Set2")
plt.title("The distribution of heart attack according to sex")
for p in ax.containers:
    ax.bar_label(p)


# In[41]:


# the average heart attack risk percentage according to sex
# 1 --> male
# 0 --> female
ds_heart.groupby('sex').attack.apply(lambda x: x.sum()/x.size * 100)


# In[42]:


plt.subplot(1,2,1)
ax = sns.countplot(x='attack', data=ds_heart, palette="Set2")
plt.title('Distribution of Attack ')
plt.xlabel('Attack')
plt.ylabel('Count')
ax.bar_label(ax.containers[0], fontsize=10, color='grey', fontweight='bold')
plt.subplot(1,2,2)
plt.pie(ds_heart.attack.value_counts(), labels = ds_heart.attack.value_counts().index, autopct = '%1.1f%%', startangle = 40, explode = (0, 0.1), colors = ['grey', 'maroon'] )
plt.title('Distribution of the Attack %')
plt.tight_layout()
plt.show()


# ## Max Heart Rate Achieved V/s Probability of Heart Attack

# In[43]:


graphanalysis("Max Heart Rate Achieved & Heart Attack Relation","maxheartrateachieved","attack")


# ## Resting Blood Pressure V/s Probability of Heart Attack

# In[44]:


graphanalysis('Resting Blood Pressure','restingbloodpressure','attack')


# ## Cholestoral Level in Blood V/s Probability of Heart Attack

# In[45]:


graphanalysis('Cholestoral','cholestoral','attack')


# In[46]:


graphanalysis('Previous peak','oldpeak','attack')


# ## Analysis other Varriable Result in Heart Attack 

# In[47]:


def graphshow(x,y,z):
    sns.countplot(data=ds_heart, x=y, hue=z)
    title=('{} Vs Attack Chances\n').format(x)
    plt.title(title)
    plt.xlabel(x)
    plt.show()


# In[48]:


graphshow('Stress Test','stresstest','attack')
graphshow('Number Of Major Vessels','numberofmajorvessels','attack')
graphshow('Chest Pain Type','chestpaintype','attack')
graphshow('Resting ECG','restingecg','attack')
graphshow('Slope','slope','attack')
graphshow('Exercise Induced Angina','exerciseinducedangina','attack')
graphshow('Fasting Blood Sugar','fastingbloodsugar','attack')


# In[49]:


ds_heart.corr()['attack'].sort_values().drop('attack').plot(kind = 'barh', color="green");
# Correlation of the target column with other columns


# In[70]:


sns.heatmap(ds_heart.corr(), annot=True)


# ### Modelling

# In[51]:


# Split 75:25
x_train=ds_heart.drop(columns=["attack"])
y_train=ds_heart["attack"]
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25)


# In[52]:


print('Train dataset shape:',x_train.shape)
print('Test dataset shape', y_train.shape)


# In[53]:


numeric_columns = [column for column in x_train.columns if (ds_heart[column].dtype == 'float64' or ds_heart[column].dtype == 'int64')]
print(numeric_columns)
print('#'*99)
categorical_columns = x_train.select_dtypes(include='object').columns
print(categorical_columns)


# In[54]:


numeric_features = Pipeline([
    ('handlingmissingvalues',SimpleImputer(strategy='median')),
    ('scaling',StandardScaler(with_mean=True))
])
print(numeric_features)


# In[55]:


categorical_features = Pipeline([
    ('handlingmissingvalues',SimpleImputer(strategy='most_frequent')),
    ('encoding', OneHotEncoder()),
    ('scaling', StandardScaler(with_mean=False))
])
print(categorical_features)


# In[56]:


processing = ColumnTransformer([
                                ('numeric', numeric_features, numeric_columns),
                                ('categorical', categorical_features, categorical_columns)
                               ])
print(processing)


# ## Model Preparation & Model Evaluation

# In[57]:


def prepare_model(algorithm):
    model = Pipeline(steps= [
                             ('processing',processing),
                             ('pca', TruncatedSVD(n_components=3, random_state=12)),
                             ('modeling', algorithm)
                            ])
    model.fit(x_train, y_train)
    return model


# In[58]:


def prepare_confusion_matrix(algo, model):
    print(algo)
    plt.figure(figsize=(6,3))
    pred = model.predict(x_test)
    cm = confusion_matrix(y_test, pred)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax)
    plt.show()
    
    # labels, title and ticks
    ax.set_xlabel('Predicted Labels');ax.set_ylabel('True Labels'); 
    ax.set_title('Confusion Matrix');


# In[59]:


def prepare_classification_report(algo, model):
    print(algo+' Report :')
    pred = model.predict(x_test)
    print(classification_report(y_test, pred))


# In[60]:


def prepare_roc_curve(algo, model):
    print(algo)
    y_pred_proba = model.predict_proba(x_test)[::,1]
    fpr, tpr, thresholds = roc_curve(y_test,  y_pred_proba)
    roc_auc = auc(fpr, tpr)
    curve = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    curve.plot()
    plt.show()


# In[61]:


algorithms = [('Random Forest calssifier', RandomForestClassifier()), 
              ('Gradientboot classifier',GradientBoostingClassifier()),
              ('XGBClassifier', XGBClassifier())
             ]


# In[62]:


trained_models = []
model_and_score = {}

for index, tup in enumerate(algorithms):
    model = prepare_model(tup[1])
    model_and_score[tup[0]] = str(model.score(x_train,y_train)*100)+"%"
    trained_models.append((tup[0],model))


# ## Evaluation Metrics

# In[63]:


print(model_and_score)


# In[64]:


for index, tup in enumerate(trained_models):
    prepare_confusion_matrix(tup[0], tup[1])


# In[65]:


for index, tup in enumerate(trained_models):
    prepare_classification_report(tup[0], tup[1])
    print("\n")


# In[66]:


print('Test dataset shape:',x_test.shape)
print('Tes dataset shape', y_test.shape)


# In[67]:


encoder = LabelEncoder()
y_test = encoder.fit_transform(y_test)

for index, tup in enumerate(trained_models):
    prepare_roc_curve(tup[0], tup[1])


# In[75]:


x = pd.DataFrame([
    [":","Random Forest calssifier",":","100",":","0.85",":","0.78"],
    [":","Gradientboot classifier",":","98.6",":","0.82",":","0.74"],
    [":","XGB Classifier",":","100",":","0.80",":","0.71"]],
    columns=[":","Model",":","Train Accuracy",":","AUC SCORE",":","f1-Score"]
)
print(x)

