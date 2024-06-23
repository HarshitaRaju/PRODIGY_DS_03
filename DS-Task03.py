#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt


# In[3]:


file_path = r'C:\Users\harsh\bank.csv'
df = pd.read_csv(file_path, sep=';')
df.head()


# In[4]:


label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])
df.head()


# In[5]:


X = df.drop(columns='y')
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[6]:


clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)


# In[13]:


y_pred = clf.predict(X_test)
print("First 10 Predictions:", y_pred[:20])
print("First 10 True Values:", y_test[:20].values)


# In[9]:


accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')


# In[10]:


joblib.dump(clf, 'decision_tree_model.pkl')


# In[11]:


for column, encoder in label_encoders.items():
    joblib.dump(encoder, f'label_encoder_{column}.pkl')


# In[21]:


plt.figure(figsize=(20,15))  
plot_tree(clf, 
          filled=True,
          feature_names=X.columns, 
          class_names=['No', 'Yes'], 
          rounded=True, 
          proportion=True,  
          precision=2, 
          fontsize=4)  
plt.show()


# In[ ]:




