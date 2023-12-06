#!/usr/bin/env python
# coding: utf-8

# In[573]:


import pandas as pd


# In[574]:


def sort_dataset(dataset_df):
    dataset_df = dataset_df.sort_values(by='year', ascending=True)
    return dataset_df


# In[575]:


def split_dataset(dataset_df):
    from sklearn.model_selection import train_test_split
    
    dataset_df['salary'] *= 0.001
    
    train = dataset_df.iloc[:1718,:]
    test = dataset_df.iloc[1718:,:]
    
    X_train = train.drop(columns="salary", axis=1)
    X_test = test.drop(columns="salary", axis=1)
    Y_train = train["salary"]
    Y_test = test["salary"]
    
    return X_train, X_test, Y_train, Y_test


# In[576]:


def extract_numerical_cols(dataset_df):
    dataset_df = dataset_df[['age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'fly', 'war']]
    return dataset_df


# In[577]:


def train_predict_decision_tree(X_train, Y_train, X_test):
    from sklearn.tree import DecisionTreeRegressor
    
    dt_cls = DecisionTreeRegressor()
    dt_cls.fit(X_train, Y_train)
    
    X_test = dt_cls.predict(X_test)
    
    return X_test


# In[578]:


def train_predict_random_forest(X_train, Y_train, X_test):
    from sklearn.ensemble import RandomForestRegressor
    
    rf_cls = RandomForestRegressor()
    rf_cls.fit(X_train, Y_train)
    
    X_test = rf_cls.predict(X_test)
    
    return X_test


# In[579]:


def train_predict_svm(X_train, Y_train, X_test):
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    
    svm_pipe = make_pipeline(StandardScaler(), SVR())
    svm_pipe.fit(X_train, Y_train)
    
    X_test = svm_pipe.predict(X_test)
    
    return X_test


# In[580]:


def calculate_RMSE(labels, predictions):
    import numpy as np
    
    res = np.sqrt(np.mean((predictions-labels)**2))
    
    return res


# In[581]:


if __name__=='__main__':
    #DO NOT MODIFY THIS FUNCTION UNLESS PATH TO THE CSV MUST BE CHANGED.
    data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')

    sorted_df = sort_dataset(data_df)
    X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)
    
    X_train = extract_numerical_cols(X_train)
    X_test = extract_numerical_cols(X_test)

    dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
    rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
    svm_predictions = train_predict_svm(X_train, Y_train, X_test)
    
    print ("Decision Tree Test RMSE: ", calculate_RMSE(Y_test, dt_predictions))
    print ("Random Forest Test RMSE: ", calculate_RMSE(Y_test, rf_predictions))
    print ("SVM Test RMSE: ", calculate_RMSE(Y_test, svm_predictions))


# In[ ]:




