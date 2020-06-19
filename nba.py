#!/usr/bin/env python
# coding: utf-8

# In[55]:


import warnings

warnings.simplefilter('ignore')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[56]:


# Run the CSV
nba_stats = pd.read_excel('../Clean data/top_8_teams_2017-2018.xlsx')
nba_stats


# In[57]:


nba_stats.drop(nba_stats.columns[nba_stats.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
nba_stats


# In[58]:


df = nba_stats[(nba_stats['Team'] == 'UTA') | (nba_stats['Opponent'] == 'SAS')]
TOR = df.iloc[:]
TOR.head(1)


# In[59]:


nba_stats.WINorLOSS[nba_stats.WINorLOSS == 'W'] = 0
nba_stats.WINorLOSS[nba_stats.WINorLOSS == 'L'] = 1
nba_stats


# In[60]:


nba_stats.Home[nba_stats.Home == 'Home'] = 0
nba_stats.Home[nba_stats.Home == 'Away'] = 1
nba_stats


# In[61]:


uniqueTeams = nba_stats['Team'].unique()
dictTeams = {}
cnt = 1
for team in uniqueTeams:
    dictTeams[team] = cnt
    cnt+=1

dictTeams

for key,value in dictTeams.items():
    print('{} = {}'.format(key,value))
    #nba_stats['Team']== key and replace Team with value
    #nba_stats['Opponent'] == key and replace Opponent with value
    


# In[62]:


dictTeams['UTA'] = 16
dictTeams


# In[63]:


nba_stats.replace(dictTeams, inplace=True)
nba_stats


# In[64]:


nba_stats.drop(nba_stats.columns[nba_stats.columns.str.contains('Date',case = False)],axis = 1, inplace = True)
nba_stats


# In[65]:


nba_stats = nba_stats.dropna(how='any')
nba_stats


# In[66]:


nba_stats.drop(nba_stats.columns[nba_stats.columns.str.contains('TeamPoints', case = False)],axis = 1, inplace = True)
nba_stats


# In[67]:


nba_stats.drop(nba_stats.columns[nba_stats.columns.str.contains('OpponentPoints',case = False)],axis = 1, inplace = True)
nba_stats


# In[68]:


X = nba_stats.drop("WINorLOSS", axis=1)
y = nba_stats["WINorLOSS"]
print(X.shape, y.shape)


# In[69]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test, = train_test_split(X, y, random_state=1, stratify=y)


# In[70]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier


# In[71]:


nba_stats.dtypes


# In[72]:


classifier.fit(X_train, y_train)


# In[73]:


print(f"Training Data Score: {classifier.score(X_train, y_train)}")
print(f"Testing Data Score: {classifier.score(X_test, y_test)}")


# In[74]:


predictions = classifier.predict(X_test)
table = pd.DataFrame({"Prediction": predictions, "Actual": y_test})
table['Correct'] = table.Prediction == table.Actual
table


# In[75]:


from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


# In[76]:


import pickle

with open('nbamodel.pkl', 'wb') as file:
    pickle.dump(classifier, file)


# In[77]:


classifier = pickle.load(open('nbamodel.pkl', 'rb'))

