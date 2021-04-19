#!/usr/bin/env python
# coding: utf-8

# # THE SPARK FOUNDATION

# # Name:Ramavath Raghu Naik

# # Prediction Using Supervised Learning

# # Predict the Percentage of an student based on the no. of Study Hours.

# In[4]:



import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn import metrics
from sklearn.linear_model import *
from sklearn.model_selection import train_test_split
import seaborn as sn
get_ipython().run_line_magic('matplotlib', 'inline')


url = "http://bit.ly/w-data"
data = pd.read_csv(url)
print("Data imported successfully")
data.head(10)


# In[4]:



import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn import metrics
from sklearn.linear_model import *
from sklearn.model_selection import train_test_split
import seaborn as sn
get_ipython().run_line_magic('matplotlib', 'inline')


url = "http://bit.ly/w-data"
data = pd.read_csv(url)
print("Data imported successfully")
data.head(10)


# In[5]:


data.describe()


# In[6]:


data.info()


# In[7]:



data.plot(x='Hours', y='Scores', style='o')
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# In[8]:



X=data.iloc[:,:-1].values
y=data.iloc[:,1].values
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=0)


# In[9]:



regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")


# In[10]:



line = regressor.coef_*X+regressor.intercept_
plt.figure(figsize=(8,6))
plt.scatter(X, y)
plt.plot(X, line);
plt.xlabel("Study Hours")
plt.ylabel("Percentage Scored")
plt.title("Line of Best Fit:-")
plt.show()


# In[11]:


print(X_test)
y_pred = regressor.predict(X_test)


# In[12]:


df = pd.DataFrame({'Hours Studied': X_test.reshape(-1,), 'Predicted Percentage': y_pred, 'Actual Percentage': y_test})  
df


# # Evaluating the Model

# In[13]:



print('Mean Absolute Error:',metrics.mean_absolute_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_absolute_error(y_test, y_pred)))
print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))


# # Predicted score for a student who studies for 9.25 hrs/day :

# In[14]:



h = 9.25
arr = np.array(h)
hours=arr.reshape(1,1)
own_pred = regressor.predict(hours)
print("No of Hours = {}".format(hours[0][0]))
print("Predicted Score = {}".format(own_pred[0]))


# final results 

# In[ ]:




